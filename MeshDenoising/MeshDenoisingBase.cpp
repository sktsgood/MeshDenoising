#include "MeshDenoisingBase.h"

void getFaceArea(MyMesh& mesh, std::vector<double>& area)
{
    area.resize(mesh.n_faces());

    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
    {
        std::vector<MyMesh::Point> point;
        point.resize(3); int index = 0;
        for (MyMesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); fv_it++)
        {
            point[index] = mesh.point(*fv_it);
            index++;
        }
        MyMesh::Point edge1 = point[1] - point[0];
        MyMesh::Point edge2 = point[1] - point[2];
        double S = 0.5 * (edge1 % edge2).length();
        area[(*f_it).idx()] = S;
    }
}

void getFaceCentroid(MyMesh& mesh, std::vector<MyMesh::Point>& centroid)
{
    centroid.resize(mesh.n_faces(), MyMesh::Point(0.0, 0.0, 0.0));
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
    {
        MyMesh::Point pt = mesh.calc_face_centroid(*f_it);
        centroid[(*f_it).idx()] = pt;
    }
}

void computeFaceNormals(MyMesh& mesh, std::vector<MyMesh::Normal>& normals, int start, int end) {
    for (int i = start; i < end; ++i) {
        MyMesh::FaceHandle face_handle = mesh.face_handle(i);
        MyMesh::Normal n = mesh.normal(face_handle);
        normals[i] = n;
    }
}

void getFaceNormal(MyMesh& mesh, std::vector<MyMesh::Normal>& normals)
{
    mesh.request_face_normals();
    mesh.update_face_normals();

    int num_faces = mesh.n_faces();
    normals.resize(num_faces);

    int num_threads = std::thread::hardware_concurrency();  // 获取可用的硬件线程数
    int block_size = (num_faces + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    // 启动多个线程，计算法向量
    for (int i = 0; i < num_threads; ++i) {
        int start = i * block_size;
        int end = std::min(start + block_size, num_faces);

        threads.emplace_back(computeFaceNormals, std::ref(mesh), std::ref(normals), start, end);
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
}

void getFaceNeighbor(MyMesh& mesh, MyMesh::FaceHandle fh, int ring, FaceNeighborType face_neighbor_type,
    std::vector<MyMesh::FaceHandle>& face_neighbor)
{
    face_neighbor.clear();
    std::set<int> visited_faces;
    std::queue<std::pair<MyMesh::FaceHandle, int>> face_queue; // Pair of FaceHandle and current ring level

    face_queue.push({ fh, 0 });
    visited_faces.insert(fh.idx());

    while (!face_queue.empty())
    {
        auto current = face_queue.front();
        face_queue.pop();

        MyMesh::FaceHandle current_fh = current.first;
        int current_ring = current.second;

        if (current_ring == ring)
        {
            face_neighbor.push_back(current_fh);
            continue;
        }

        if (face_neighbor_type == kEdgeBased)
        {
            for (MyMesh::FaceFaceIter ff_it = mesh.ff_iter(current_fh); ff_it.is_valid(); ff_it++)
            {
                if (visited_faces.find(ff_it->idx()) == visited_faces.end())
                {
                    face_queue.push({ *ff_it, current_ring + 1 });
                    visited_faces.insert(ff_it->idx());
                }
            }
        }
        else if (face_neighbor_type == kVertexBased)
        {
            for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(current_fh); fv_it.is_valid(); fv_it++)
            {
                for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*fv_it); vf_it.is_valid(); vf_it++)
                {
                    if ((*vf_it) != fh && visited_faces.find(vf_it->idx()) == visited_faces.end())
                    {
                        face_queue.push({ *vf_it, current_ring + 1 });
                        visited_faces.insert(vf_it->idx());
                    }
                }
            }
        }
    }
}


void getFaceIter(MyMesh& mesh, std::vector<MyMesh::FaceIter>& face_iter)
{
    for (MyMesh::FaceIter fh = mesh.faces_begin(); fh != mesh.faces_end(); ++fh) {
        int f_id = fh->idx(); // 获取当前面的索引 f_id
        face_iter[f_id] = fh;
    }
}

double computeSigmaCenter(MyMesh mesh, std::vector<MyMesh::Point> FaceCenter, double multiple)
{
    double SigmaCenter = 0.0;
    double num = 0.0;
    for (MyMesh::FaceIter fh = mesh.faces_begin(); fh != mesh.faces_end(); ++fh) {
        int f_id = fh->idx(); // 获取当前面的索引 f_id
        for (MyMesh::FaceFaceIter nei_fh = mesh.ff_iter(*fh); nei_fh.is_valid(); ++nei_fh) {
            int ff_id = nei_fh->idx(); // 获取相邻面的索引 ff_id
            SigmaCenter += (FaceCenter[f_id] - FaceCenter[ff_id]).norm();// 获取当前面和相邻面的中心点，并计算其差的模并累加到 SigmaCenter
            num++;
        }
    }
    return SigmaCenter = SigmaCenter * multiple / num;	// 计算平均值并归一化
}

void computeAdjustedFaceNormals(MyMesh& mesh, const std::vector<MyMesh::Normal>& FaceNormal,
    std::vector<MyMesh::Normal>& NewFaceNormal, const std::vector<double>& FaceArea,
    const std::vector<MyMesh::Point>& FaceCenter, double SigmaCenter, double SigmaNormal,
    const std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor,
    int start, int end) {
    for (int i = start; i < end; ++i) {
        double Kp = 0;
        MyMesh::Normal NewN(0, 0, 0);
        int fh_id = i;
        const std::vector<MyMesh::FaceHandle>& face_neighbor = all_face_neighbor[fh_id];

        for (const auto& neighbor : face_neighbor) {
            int nei_fh_id = neighbor.idx();
            double delta_center = (FaceCenter[fh_id] - FaceCenter[nei_fh_id]).norm();
            double delta_normal = (FaceNormal[fh_id] - FaceNormal[nei_fh_id]).norm();
            double Aj = FaceArea[nei_fh_id];
            double Wc = exp(-delta_center * delta_center / (2 * SigmaCenter * SigmaCenter));
            double Ws = exp(-delta_normal * delta_normal / (2 * SigmaNormal * SigmaNormal));
            NewN += Aj * Wc * Ws * FaceNormal[nei_fh_id];
            Kp += Aj * Wc * Ws;
        }

        if (face_neighbor.empty()) {
            NewFaceNormal[fh_id] = FaceNormal[fh_id];
        }
        else {
            NewFaceNormal[fh_id] = NewN / Kp;
        }
        NewFaceNormal[fh_id].normalize();
    }
}

void adjustFaceNormals(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal,
    std::vector<MyMesh::Normal>& NewFaceNormal, std::vector<double>& FaceArea,
    std::vector<MyMesh::Point>& FaceCenter, double SigmaCenter, double SigmaNormal,
    std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor) {
    int num_faces = mesh.n_faces();
    NewFaceNormal.resize(num_faces);

    int num_threads = std::thread::hardware_concurrency();
    int block_size = (num_faces + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    // 启动多个线程，处理面片块
    for (int i = 0; i < num_threads; ++i) {
        int start = i * block_size;
        int end = std::min(start + block_size, num_faces);

        threads.emplace_back(computeAdjustedFaceNormals, std::ref(mesh), std::cref(FaceNormal),
            std::ref(NewFaceNormal), std::cref(FaceArea), std::cref(FaceCenter),
            SigmaCenter, SigmaNormal, std::cref(all_face_neighbor), start, end);
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
}

//void adjustFaceNormals(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal, 
//    std::vector<MyMesh::Normal>& NewFaceNormal, std::vector<double>& FaceArea, 
//    std::vector<MyMesh::Point>& FaceCenter, double SigmaCenter, double SigmaNormal, 
//    std::vector<std::vector<MyMesh::FaceHandle> > all_face_neighbor)
//{
//    for (MyMesh::FaceIter fh = mesh.faces_begin(); fh != mesh.faces_end(); ++fh)
//    {
//        double Kp = 0;
//        MyMesh::Normal NewN(0, 0, 0);
//        int fh_id = fh->idx();// 获取当前面的索引 fh_id
//        const std::vector<MyMesh::FaceHandle> face_neighbor = all_face_neighbor[fh_id];
//        for (int j = 0; j < (int)face_neighbor.size(); j++)
//        {
//            int nei_fh_id = face_neighbor[j].idx();// 获取邻面的索引 nei_fh_id
//            double delta_center = (FaceCenter[fh_id] - FaceCenter[nei_fh_id]).norm();//||ci-cj||
//            double delta_normal = (FaceNormal[fh_id] - FaceNormal[nei_fh_id]).norm();//||ni-nj||
//            double Aj = FaceArea[nei_fh_id];
//            double Wc = exp(-delta_center * delta_center / (2 * SigmaCenter * SigmaCenter));
//            double Ws = exp(-delta_normal * delta_normal / (2 * SigmaNormal * SigmaNormal));
//            NewN += Aj * Wc * Ws * FaceNormal[nei_fh_id];
//            Kp += Aj * Wc * Ws;
//        }
//        if (!face_neighbor.size())
//            NewFaceNormal[fh_id] = FaceNormal[fh_id];
//        else
//            NewFaceNormal[fh_id] = NewN / Kp;
//        NewFaceNormal[fh_id].normalize();
//
//    }
//
//}

void adjustVertexPositions(MyMesh& mesh, std::vector<MyMesh::Normal>& filtered_normals, 
    std::vector<MyMesh::Point>& Point, int PointNum, bool fixed_boundary)
{
    std::vector<MyMesh::Point> new_points(mesh.n_vertices());

    std::vector<MyMesh::Point> centroid;

    for (int iter = 0; iter < PointNum; iter++)
    {
        getFaceCentroid(mesh, centroid);
        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
        {
            MyMesh::Point p = mesh.point(*v_it);
            if (fixed_boundary && mesh.is_boundary(*v_it))
            {
                new_points.at(v_it->idx()) = p;
            }
            else
            {
                double face_num = 0.0;
                MyMesh::Point temp_point(0.0, 0.0, 0.0);
                for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); vf_it++)
                {
                    MyMesh::Normal temp_normal = filtered_normals[vf_it->idx()];
                    MyMesh::Point temp_centroid = centroid[vf_it->idx()];
                    temp_point += temp_normal * (temp_normal | (temp_centroid - p));
                    face_num++;
                }
                p += temp_point / face_num;

                new_points.at(v_it->idx()) = p;
            }
        }

        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
            mesh.set_point(*v_it, new_points[v_it->idx()]);
    }
}

void adjustNonFeatureVertexPositions(MyMesh& mesh, std::vector<MyMesh::Normal>& filtered_normals,
    std::vector<MyMesh::Point>& Point, int PointNum, bool fixed_boundary, std::vector<int>& VertexClassify, 
    std::vector<MyMesh::Normal>& VertexNormal)
{
    std::vector<MyMesh::Point> new_points(mesh.n_vertices());

    std::vector<MyMesh::Point> centroid;

    for (int iter = 0; iter < PointNum; iter++)
    {
        getFaceCentroid(mesh, centroid);
        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
        {
            MyMesh::Point p = mesh.point(*v_it);
            if (fixed_boundary && mesh.is_boundary(*v_it) || VertexClassify[v_it->idx()] != 0)
            {
                new_points.at(v_it->idx()) = p;
            }
            else
            {

                double face_num = 0.0;
                MyMesh::Point temp_point(0.0, 0.0, 0.0);
                for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); vf_it++)
                {
                    MyMesh::Normal temp_normal = filtered_normals[vf_it->idx()];
                    MyMesh::Point temp_centroid = centroid[vf_it->idx()];
                    if (temp_normal.dot(VertexNormal[v_it->idx()]) > 0.6)
                    {
                        temp_point += temp_normal * (temp_normal | (temp_centroid - p));
                        face_num++;
                    }
                }
                if(face_num != 0) p += temp_point / face_num;
                new_points.at(v_it->idx()) = p;
            }
        }

        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
            mesh.set_point(*v_it, new_points[v_it->idx()]);
    }
}

