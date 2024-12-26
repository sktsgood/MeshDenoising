#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <chrono>
#include <cmath>
#include "MeshDenoisingBase.h"
#include "FeatureVertexMeshDenoising.h"
// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
using namespace OpenMesh;

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Vector3d Vector3d;

#define MAX_DOUBLE 1000000000

struct MS {
    double ms;//相似度
    int fh_id;//面id
    MS() : ms(0), fh_id(0) {}
    MS(double ms, int fh_id) : ms(ms), fh_id(fh_id) {}

    // 小根堆需要的比较函数
    bool operator>(const MS& other) const {
        return ms > other.ms;
    }
};

void getALLX_RingFaceNeighbor(MyMesh& mesh, FaceNeighborType face_neighbor_type, bool include_central_face, std::vector<std::vector<MyMesh::FaceHandle> >& all_face_neighbor,int ring)
{
    std::vector<MyMesh::FaceHandle> face_neighbor;
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
    {
        if (face_neighbor_type == kVertexBased)
            getFaceNeighbor(mesh, *f_it, ring, kVertexBased, face_neighbor);
        else if (face_neighbor_type == kEdgeBased)
            getFaceNeighbor(mesh, *f_it, ring, kEdgeBased, face_neighbor);
        if (include_central_face)
            face_neighbor.push_back(*f_it);
        all_face_neighbor[f_it->idx()] = face_neighbor;
    }
}

void getAllGuidedNeighbor(MyMesh& mesh, std::vector<std::vector<MyMesh::FaceHandle> >& all_guided_neighbor)
{
    std::vector<MyMesh::FaceHandle> face_neighbor;
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
    {
        getFaceNeighbor(mesh, *f_it, 1, kVertexBased, face_neighbor);
        face_neighbor.push_back(*f_it);
        all_guided_neighbor[f_it->idx()] = face_neighbor;
    }
}

void getFaceNeighborInnerEdge(MyMesh& mesh, const std::vector<MyMesh::FaceHandle>& face_neighbor, std::vector<MyMesh::EdgeHandle>& inner_edge)
{
    inner_edge.clear();
    std::vector<bool> edge_flag((int)mesh.n_edges(), false);
    std::vector<bool> face_flag((int)mesh.n_faces(), false);

    for (const auto& face : face_neighbor) {
        face_flag[face.idx()] = true;
    }

    for (const auto& face : face_neighbor) {
        for (MyMesh::FaceEdgeIter fe_it = mesh.fe_iter(face); fe_it.is_valid(); fe_it++) {
            if ((!edge_flag[fe_it->idx()]) && (!mesh.is_boundary(*fe_it))) {
                edge_flag[fe_it->idx()] = true;
                MyMesh::HalfedgeHandle heh = mesh.halfedge_handle(*fe_it, 0);
                MyMesh::FaceHandle f = mesh.face_handle(heh);
                MyMesh::HalfedgeHandle heho = mesh.opposite_halfedge_handle(heh);
                MyMesh::FaceHandle fo = mesh.face_handle(heho);
                if (face_flag[f.idx()] && face_flag[fo.idx()]) {
                    inner_edge.push_back(*fe_it);
                }
            }
        }
    }
}

void computeRangeAndMeanNormal(MyMesh& mesh, const std::vector<std::vector<MyMesh::FaceHandle>>& all_guided_neighbor,
    const std::vector<MyMesh::Normal>& normals, std::vector<double>& range_and_mean_normal,
    int start, int end) {
    const double epsilon = 1.0e-9;

    for (int i = start; i < end; ++i) {
        int index = i;
        const std::vector<MyMesh::FaceHandle>& face_neighbor = all_guided_neighbor[index];
        double metric = 0.0;
        double maxdiff = -1.0;

        // 计算相邻面片的法向量差值
        for (int i = 0; i < (int)face_neighbor.size(); i++) {
            int index_i = face_neighbor[i].idx();
            MyMesh::Normal ni = normals[index_i];

            for (int j = i + 1; j < (int)face_neighbor.size(); j++) {
                int index_j = face_neighbor[j].idx();
                MyMesh::Normal nj = normals[index_j];
                double diff = (ni - nj).length();

                if (diff > maxdiff) {
                    maxdiff = diff;
                }
            }
        }

        // 获取内部边缘并计算相邻面片法向量差值
        std::vector<MyMesh::EdgeHandle> inner_edge_handle;
        getFaceNeighborInnerEdge(mesh, face_neighbor, inner_edge_handle);
        double sum_tv = 0.0, max_tv = -1.0;
        for (const auto& edge : inner_edge_handle) {
            MyMesh::HalfedgeHandle heh = mesh.halfedge_handle(edge, 0);
            MyMesh::FaceHandle f = mesh.face_handle(heh);
            MyMesh::Normal n1 = normals[f.idx()];
            MyMesh::HalfedgeHandle heho = mesh.opposite_halfedge_handle(heh);
            MyMesh::FaceHandle fo = mesh.face_handle(heho);
            MyMesh::Normal n2 = normals[fo.idx()];
            double current_tv = (n1 - n2).length();
            max_tv = std::max(max_tv, current_tv);
            sum_tv += current_tv;
        }
        metric = maxdiff * max_tv / (sum_tv + epsilon);
        range_and_mean_normal[index] = metric;
    }
}

void getRangeAndMeanNormal(MyMesh& mesh, const std::vector<std::vector<MyMesh::FaceHandle>>& all_guided_neighbor,
    const std::vector<MyMesh::Normal>& normals, std::vector<double>& range_and_mean_normal) {
    int num_faces = mesh.n_faces();
    range_and_mean_normal.resize(num_faces);

    int num_threads = std::thread::hardware_concurrency();
    int block_size = (num_faces + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    // 启动多线程，处理每个线程负责的一部分面片
    for (int i = 0; i < num_threads; ++i) {
        int start = i * block_size;
        int end = std::min(start + block_size, num_faces);
        threads.emplace_back(computeRangeAndMeanNormal, std::ref(mesh), std::ref(all_guided_neighbor),
            std::ref(normals), std::ref(range_and_mean_normal), start, end);
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
}

void computeGuidedNormals(MyMesh& mesh, const std::vector<double>& FaceArea, const std::vector<MyMesh::Normal>& FaceNormal,
    std::vector<MyMesh::Normal>& guided_normals, const std::vector<double>& range_and_mean_normal,
    const std::vector<std::vector<MyMesh::FaceHandle>>& all_guided_neighbor, double p,
    int start, int end) {
    for (int i = start; i < end; ++i) {
        MyMesh::FaceHandle face = mesh.face_handle(i);
        MyMesh::Normal average_normal(0.0, 0.0, 0.0);
        MyMesh::Normal ni = FaceNormal[face.idx()];
        const std::vector<MyMesh::FaceHandle>& face_neighbor = all_guided_neighbor[face.idx()];
        double min_range = 1.0e8;
        int min_idx = 0;

        // 找到最小范围的邻居
        for (int j = 0; j < (int)face_neighbor.size(); ++j) {
            double current_range = range_and_mean_normal[face_neighbor[j].idx()];
            if (min_range > current_range) {
                min_range = current_range;
                min_idx = j;
            }
        }

        MyMesh::FaceHandle min_face_handle = face_neighbor[min_idx];
        const std::vector<MyMesh::FaceHandle>& min_face_neighbor = all_guided_neighbor[min_face_handle.idx()];

        // 计算加权法向量
        for (const auto& neighbor_face : min_face_neighbor) {
            int index_i = neighbor_face.idx();
            double area_weight = FaceArea[index_i];
            MyMesh::Normal nj = FaceNormal[index_i];
            double dot_product = dot(ni, nj);

            if (dot_product > p) {
                average_normal += nj * area_weight;
            }
        }

        average_normal.normalize();
        guided_normals[face.idx()] = average_normal;
    }
}

void getGuidedNormals(MyMesh& mesh, std::vector<double>& FaceArea, std::vector<MyMesh::Normal>& FaceNormal, std::vector<MyMesh::Normal>& guided_normals, std::vector<double> range_and_mean_normal, std::vector<std::vector<MyMesh::FaceHandle> >& all_guided_neighbor, double p)
{
    getRangeAndMeanNormal(mesh, all_guided_neighbor, FaceNormal, range_and_mean_normal);
    int num_faces = mesh.n_faces();
    guided_normals.resize(num_faces);

    int num_threads = std::thread::hardware_concurrency();
    int block_size = (num_faces + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    // 启动多线程处理面片范围
    for (int i = 0; i < num_threads; ++i) {
        int start = i * block_size;
        int end = std::min(start + block_size, num_faces);

        threads.emplace_back(computeGuidedNormals, std::ref(mesh), std::ref(FaceArea), std::ref(FaceNormal),
            std::ref(guided_normals), std::cref(range_and_mean_normal),
            std::cref(all_guided_neighbor), p, start, end);
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
}

//引导双边滤波
void computeAdjustedNormals(MyMesh& mesh, const std::vector<MyMesh::Normal>& Initial_Normal,
    const std::vector<MyMesh::Normal>& GuidedNormal, std::vector<MyMesh::Normal>& NewFaceNormal,
    const std::vector<double>& FaceArea, const std::vector<MyMesh::Point>& FaceCenter,
    double SigmaCenter, double SigmaNormal, const std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor,
    int start, int end) {
    for (int i = start; i < end; ++i) {
        MyMesh::FaceHandle fh = mesh.face_handle(i);
        MyMesh::Normal NewN(0, 0, 0);
        int fh_id = fh.idx();  // 当前面索引
        const std::vector<MyMesh::FaceHandle>& face_neighbor = all_face_neighbor[fh_id];

        for (const auto& neighbor : face_neighbor) {
            int nei_fh_id = neighbor.idx();  // 获取邻面的索引
            double delta_center = (FaceCenter[fh_id] - FaceCenter[nei_fh_id]).length();
            double delta_normal = (GuidedNormal[fh_id] - GuidedNormal[nei_fh_id]).length();
            double Aj = FaceArea[nei_fh_id];
            double Wc = exp(-delta_center * delta_center / (2 * SigmaCenter * SigmaCenter));
            double Ws = exp(-delta_normal * delta_normal / (2 * SigmaNormal * SigmaNormal));
            NewN += Aj * Wc * Ws * Initial_Normal[nei_fh_id];
        }

        // 归一化并保存结果
        if (!face_neighbor.empty()) {
            NewFaceNormal[fh_id] = NewN.normalize();
        }
    }
}

void adjustGuidedNormals(MyMesh& mesh, std::vector<MyMesh::Normal>& Initial_Normal, std::vector<MyMesh::Normal>& GuidedNormal,
    std::vector<MyMesh::Normal>& NewFaceNormal, std::vector<double>& FaceArea,
    std::vector<MyMesh::Point>& FaceCenter, double SigmaCenter, double SigmaNormal,
    std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor) {
    int num_faces = mesh.n_faces();
    NewFaceNormal.resize(num_faces);

    int num_threads = std::thread::hardware_concurrency();
    int block_size = (num_faces + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    // 启动多线程，计算调整后的法向量
    for (int i = 0; i < num_threads; ++i) {
        int start = i * block_size;
        int end = std::min(start + block_size, num_faces);

        threads.emplace_back(computeAdjustedNormals, std::ref(mesh), std::cref(Initial_Normal), std::cref(GuidedNormal),
            std::ref(NewFaceNormal), std::cref(FaceArea), std::cref(FaceCenter),
            SigmaCenter, SigmaNormal, std::cref(all_face_neighbor), start, end);
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
}

//void adjustGuidedNormals_two(MyMesh& mesh, std::vector<MyMesh::Normal>& Initial_Normal, std::vector<MyMesh::Normal>& GuidedNormal, 
//    std::vector<MyMesh::Normal>& NewFaceNormal, std::vector<double>& FaceArea, std::vector<MyMesh::Point>& FaceCenter, 
//    double SigmaCenter, double SigmaNormal, double SigmaMS, std::vector<std::vector<MyMesh::FaceHandle> >& all_face_neighbor, 
//    std::vector<std::priority_queue<MS, std::vector<MS>, std::greater<MS>>>& MinSimilarity, std::vector<int>& VertexClassify)
//{
//    for (MyMesh::FaceIter fh = mesh.faces_begin(); fh != mesh.faces_end(); ++fh)
//    {
//        MyMesh::Normal NewN(0, 0, 0);
//        int fh_id = fh->idx();// 获取当前面的索引 fh_id
//        const std::vector<MyMesh::FaceHandle> face_neighbor = all_face_neighbor[fh_id];
//        for (int j = 0; j < (int)face_neighbor.size(); j++)
//        {
//            int nei_fh_id = face_neighbor[j].idx();// 获取邻面的索引 nei_fh_id
//            double delta_center = (FaceCenter[fh_id] - FaceCenter[nei_fh_id]).length();//||ci-cj||
//            double delta_normal = (GuidedNormal[fh_id] - GuidedNormal[nei_fh_id]).length();//||ni-nj||
//            double Aj = FaceArea[nei_fh_id];
//            double Wc = exp(-delta_center * delta_center / (2 * SigmaCenter * SigmaCenter));
//            double Ws = exp(-delta_normal * delta_normal / (2 * SigmaNormal * SigmaNormal));
//            NewN += Aj * Wc * Ws * Initial_Normal[nei_fh_id];
//        }
//        if (VertexClassify[fh_id] != 0)
//        {
//            for (int i = 0; i < 5; i++)// 选取前5个最相似的邻面
//            {
//                int nei_fh_id = MinSimilarity[fh_id].top().fh_id;
//                double delta_MS = MinSimilarity[fh_id].top().ms;
//                double Ak = FaceArea[nei_fh_id];
//                double sp = 0.5;
//                double Wm = exp(-delta_MS * delta_MS / (2 * SigmaMS * SigmaMS));
//                NewN += Ak * Wm * sp * Initial_Normal[nei_fh_id];
//                MinSimilarity[fh_id].pop();
//            }
//        }
//        if (face_neighbor.size())
//            NewFaceNormal[fh_id] = NewN.normalize();
//    }
//
//}

MyMesh::Point computeFaceCentroid(MyMesh& mesh, MyMesh::FaceHandle fh) {
    MyMesh::Point centroid(0, 0, 0);
    int vertex_count = 0;

    for (MyMesh::FaceVertexIter fv_it = mesh.fv_iter(fh); fv_it.is_valid(); ++fv_it) {
        centroid += mesh.point(*fv_it);
        ++vertex_count;
    }

    if (vertex_count > 0) {
        centroid /= vertex_count;
    }

    return centroid;
}

double computeFaceDistance(MyMesh& mesh1, std::vector<MyMesh::Normal>& Initial_Normal, std::vector<double>& FaceArea, MyMesh::FaceHandle fh1, MyMesh::FaceHandle fh2,const MyMesh::Point& translation_vector) {
    MyMesh::Point c1 = computeFaceCentroid(mesh1, fh1);
    MyMesh::Point c2 = computeFaceCentroid(mesh1, fh2) + translation_vector;
    MyMesh::Normal n1 = Initial_Normal[fh1.idx()];
    MyMesh::Normal n2 = Initial_Normal[fh2.idx()];
    double a1 = FaceArea[fh1.idx()];

    double distance = (c1 - c2).length() * (1 + (n1 | n2)) * a1;
    return distance;
}

//相似度计算
void computeSimilarity(MyMesh& mesh, std::vector<MyMesh::Normal>& Initial_Normal, std::vector<double>& FaceArea, 
    std::vector<std::vector<MyMesh::FaceHandle> >& all_face_neighbor, MyMesh::FaceHandle fh, 
    std::vector<MyMesh::FaceHandle>& choosed_faces,std::vector<MS>& S)
{
    int fh_id = fh.idx();
    std::vector<MyMesh::FaceHandle> p1 = all_face_neighbor[fh_id];
    for (const auto& face : choosed_faces)
    {
        if (face == fh) continue;
        std::vector<MyMesh::FaceHandle> p2 = all_face_neighbor[face.idx()];
        MyMesh::Point centroid1 = computeFaceCentroid(mesh, fh);
        MyMesh::Point centroid2 = computeFaceCentroid(mesh, face);
        MyMesh::Point translation_vector = centroid1 - centroid2;
        double total_distance = 0;
        double total_area = 0;
        for (int i = 0; i < p1.size(); i++){
            double mindistance = 10000;
            for (int j = 0; j < p2.size(); j++) {
                double distance = computeFaceDistance(mesh, Initial_Normal, FaceArea, p1[i], p2[j], translation_vector);
                if (distance < mindistance) {
                    mindistance = distance;
                }
            }
            total_distance += mindistance;
            total_area += FaceArea[p1[i].idx()];
        }
        MS hj(total_distance / total_area, face.idx());
        S.push_back(hj);
    }
}

void computeMS(MyMesh& mesh, std::vector<MyMesh::Normal>& Initial_Normal, std::vector<double>& FaceArea,
    std::vector<int>& VertexClassify, std::vector<std::priority_queue<MS, std::vector<MS>, std::greater<MS> > >& MinSimilarity)
{
    std::set<MyMesh::FaceHandle> unique_faces;
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
        int vh_id = vh->idx();
        if (VertexClassify[vh_id] != 0) {
            for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*vh); vf_it.is_valid(); ++vf_it) {
                unique_faces.insert(*vf_it);
            }
        }
    }
    std::vector<MyMesh::FaceHandle> choosed_faces;
    std::vector<std::vector<MyMesh::FaceHandle> > all_face_neighbor;
    std::vector<std::vector<std::vector<MS>>> kl;
    for (const auto& face_idx : unique_faces) {
        MyMesh::FaceHandle face = face_idx;
        choosed_faces.push_back(face);
    }
    std::vector<MyMesh::FaceHandle> choosed_face = choosed_faces;
    for (int N = 2; N <= 4; N++) {
        getALLX_RingFaceNeighbor(mesh, kVertexBased, true, all_face_neighbor, N);
        for (int i = 0; i < choosed_face.size(); i++) {
            int fh_id = choosed_face[i].idx();
            MyMesh::FaceHandle fh = choosed_face[i];
            std::vector<MS> S;
            computeSimilarity(mesh, Initial_Normal, FaceArea, all_face_neighbor, fh, choosed_face, S);
            kl[fh_id].push_back(S);
        }
    }
    for (const auto& face : choosed_face) {
        int fh_id = face.idx();
        std::vector<MS> S2 = kl[fh_id][0];
        std::vector<MS> S3 = kl[fh_id][1];
        std::vector<MS> S4 = kl[fh_id][2];
        std::priority_queue<MS, std::vector<MS>, std::greater<MS>> MinS;
        for (int i = 0; i < S2.size(); i++) {
            double MS = S2[i].ms + 0.2 * S3[i].ms + 0.05 * S4[i].ms;
            MinS.emplace(MS, S2[i].fh_id);
        }
        MinSimilarity[fh_id] = MinS;
    }

}
// 自定义哈希函数，用于 unordered_map 中存储顶点坐标
struct PointHasher {
    size_t operator()(const MyMesh::Point& p) const {
        size_t hash = 0;
        for (int i = 0; i < 3; ++i) {
            hash ^= std::hash<float>{}(p[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

void add_face_to_julei_mesh(MyMesh& mesh, MyMesh& julei_mesh,
    const std::vector<MyMesh::VertexHandle>& face_vhandles,
    std::unordered_map<MyMesh::Point, MyMesh::VertexHandle, PointHasher>& vertex_map) {
    std::vector<MyMesh::VertexHandle> julei_face_vhandles;

    // 遍历面片的顶点
    for (const auto& vh : face_vhandles) {
        MyMesh::Point point = mesh.point(vh);  // 获取顶点的坐标

        // 检查顶点是否已经存在于 julei_mesh 中
        auto it = vertex_map.find(point);
        MyMesh::VertexHandle target_vh;

        if (it != vertex_map.end()) {
            // 如果顶点已经存在，使用已有的 VertexHandle
            target_vh = it->second;
        }
        else {
            // 如果顶点不存在，添加到 julei_mesh 中
            target_vh = julei_mesh.add_vertex(point);
            vertex_map[point] = target_vh;  // 缓存该顶点的句柄
        }

        julei_face_vhandles.push_back(target_vh);
    }

    // 添加面片到 julei_mesh
    if (!julei_mesh.add_face(julei_face_vhandles).is_valid()) {
        std::cerr << "Error: Failed to add face to julei_mesh!" << std::endl;
    }
}

int main() {
    MyMesh mesh;
    MyMesh mesh1;
    MyMesh mesh2;
    MyMesh mesh3;
    //std::string mesh_path = "F:/c++ workplace/GCN-Denoiser-master/models/fertility_gaus_n3.obj";
    //std::string mesh_path = "F:/c++ workplace/MeshDenoising/Original.obj"; 
    //std::string mesh_path = "F:/c++ workplace/GuidedDenoising-master/models/Julius0.2/Noisy.obj";
    std::string mesh_path = "F:/c++ workplace/Mesh/train/Noisy/cc0.5.ply";
    //td::string mesh_path = "F:/c++ workplace/GuidedDenoising-master/models/SharpSphere0.3/Noisy.obj";
    //std::string mesh_path = "F:/c++ workplace/Mesh/train/original/casting.obj";
    // 从文件加载网格数据
    try
    {
        if (!IO::read_mesh(mesh, mesh_path))
        {
            std::cerr << "Cannot read mesh from file " << mesh_path << std::endl;
            return 1;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
        return 1;
    }
    double SigmaNormal1 = 0.35;
    double SigmaNormal2 = 0.3;
    double Sigma_S = 1.0;
    //double SigmaMS = 0.7;
    int NormalNum, PointNum;//迭代次数
    double smoothness1 = 0.01, smoothness2 = 0.01;
    double VertexSmoothness1 = 0.01, VertexSmoothness2 = 0.01;
    double Tvk = 0.05;
    double p = 0.0;//引导法线选取阈值
    bool include_central_face = true;
    FaceNeighborType face_neighbor_type = kVertexBased;
    std::cout << "请输入想要迭代法向的次数：" << std::endl;
    std::cin >> NormalNum;
    std::cout << "请输入想要迭代顶点的次数：" << std::endl;
    std::cin >> PointNum;
    auto start = std::chrono::high_resolution_clock::now(); //记录开始时间
    mesh.request_face_normals(); // 请求存储面法线的属性
    mesh.update_face_normals();  // 计算并存储所有面的法线
    std::vector<MyMesh::Normal> FaceNormal(mesh.n_faces());
    std::vector<MyMesh::Normal> NewFaceNormal(mesh.n_faces());
    std::vector<MyMesh::Normal> guided_normals(mesh.n_faces());
    std::vector<MyMesh::Normal> FaceNormal_xihua(mesh.n_faces());//细化后的面法向
    std::vector<MyMesh::Normal> Initial_Normal(mesh.n_faces());
    std::vector<double> FaceArea(mesh.n_faces());
    std::vector<MyMesh::Point> FaceCenter(mesh.n_faces());
    std::vector<MyMesh::FaceIter> Face(mesh.n_faces());
    std::vector<int> VertexClassify(mesh.n_vertices());//分类顶点的种类，其中角顶点为2，边顶点为1，面顶点为0, 圆角顶点为3
    std::vector<double> VertexS(mesh.n_vertices());//记录特征顶点的特征强度
    std::vector<Vector3d> VertexEigen(mesh.n_vertices());//记录所有顶点的特征向量
    std::vector<MyMesh::Normal> VertexNormal(mesh.n_vertices());
    std::vector<Group> FeatureVertexGroups(mesh.n_vertices());
    std::vector<std::vector<Eigen::Vector4d>> FitPlane(mesh.n_vertices());
    std::vector<MyMesh::Point> Point(mesh.n_vertices());
    std::vector<double> range_and_mean_normal(mesh.n_faces());
    std::vector<std::vector<MyMesh::FaceHandle> > all_face_neighbor((int)mesh.n_faces());
    std::vector<std::vector<MyMesh::FaceHandle> > all_guided_neighbor((int)mesh.n_faces());
    std::vector<std::priority_queue<MS, std::vector<MS>, std::greater<MS>>> MinSimilarity((int)mesh.n_faces());
    
    getAllGuidedNeighbor(mesh, all_guided_neighbor);
    getALLX_RingFaceNeighbor(mesh, face_neighbor_type, include_central_face, all_face_neighbor, 1);

    getFaceNormal(mesh, FaceNormal);
    getFaceIter(mesh, Face);
    auto end1 = std::chrono::high_resolution_clock::now();  // 记录结束时间
    std::chrono::duration<double> execution_time1 = end1 - start;  // 计算运行时间
    std::cout << "预处理运行时间: " << execution_time1.count() << " 秒" << std::endl;
    for (int idex = 0; idex < NormalNum; idex++)
    {
        getFaceNormal(mesh, Initial_Normal);
        getFaceArea(mesh, FaceArea);
        getFaceCentroid(mesh, FaceCenter);
        double SigmaCenter = computeSigmaCenter(mesh, FaceCenter, Sigma_S);//计算所有邻面中心差值的平均值作为高斯函数参数
        getGuidedNormals(mesh, FaceArea, Initial_Normal, guided_normals, range_and_mean_normal, all_guided_neighbor,p);
        adjustGuidedNormals(mesh, Initial_Normal, guided_normals, FaceNormal, FaceArea, FaceCenter, SigmaCenter, SigmaNormal1, all_face_neighbor);
        //adjustFaceNormals(mesh, Initial_Normal, NewFaceNormal, FaceArea, FaceCenter, SigmaCenter, SigmaNormal1, all_face_neighbor);

        //auto end2 = std::chrono::high_resolution_clock::now();  // 记录结束时间
        //std::chrono::duration<double> execution_time2 = end2 - end1;  // 计算运行时间
        //std::cout << "法线初步估计运行时间: " << execution_time2.count() << " 秒" << std::endl;

        //tensorVoting(mesh, FaceNormal, FaceArea, FaceCenter, VertexClassify, Tvk, VertexS, VertexEigen);
        //IsolatedFeatureElimination(mesh, VertexClassify);
        //PseudoCornerFeatureElimination(mesh, VertexClassify, VertexS);
        //WeakFeaturePointRecognition(mesh, VertexClassify, VertexEigen);
        //RoundedFeaturePointFiltering(mesh, VertexClassify, VertexEigen);

        //auto end3 = std::chrono::high_resolution_clock::now();  // 记录结束时间
        //std::chrono::duration<double> execution_time3 = end3 - end2;  // 计算运行时间
        //std::cout << "特征点识别运行时间: " << execution_time3.count() << " 秒" << std::endl;

        //clusterVertex(mesh, FaceNormal, FaceArea, FaceCenter, VertexClassify, Face, VertexNormal, FeatureVertexGroups, FitPlane);

        //auto end4 = std::chrono::high_resolution_clock::now();  // 记录结束时间
        //std::chrono::duration<double> execution_time4 = end4 - end3;  // 计算运行时间
        //std::cout << "面片聚类运行时间: " << execution_time4.count() << " 秒" << std::endl;

        //for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
        //{
        //    int i = v_it->idx();
        //    if (VertexClassify[i] == 3)
        //    {
        //        for (MyMesh::VertexFaceIter vf_it = mesh.vf_iter(*v_it); vf_it.is_valid(); vf_it++)
        //        {
        //            FaceNormal[vf_it->idx()] = NewFaceNormal[vf_it->idx()];
        //        }
        //    }
        //}
        //FacetNormalFieldRefinement(mesh, smoothness1, smoothness2, SigmaCenter, SigmaNormal2, FaceNormal, FaceNormal_xihua, FaceArea, FaceCenter, VertexClassify, Face, VertexNormal, FeatureVertexGroups, all_face_neighbor);

        //auto end5 = std::chrono::high_resolution_clock::now();  // 记录结束时间
        //std::chrono::duration<double> execution_time5 = end5 - end4;  // 计算运行时间
        //std::cout << "面片再运算运行时间: " << execution_time5.count() << " 秒" << std::endl;

        //VertexPositionUpdate(mesh, Point, FaceCenter, FaceNormal_xihua, VertexNormal, VertexClassify, VertexSmoothness1, VertexSmoothness2);
        adjustVertexPositions(mesh, FaceNormal, Point, PointNum, true);
        //adjustFeatureVertexPositions(mesh, FaceNormal_xihua, Point, PointNum, true, VertexClassify, FeatureVertexGroups, FitPlane);
    }
    
    
    

    

    std::cout << "OK" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();  // 记录结束时间
    std::chrono::duration<double> execution_time = end - start;  // 计算运行时间
    std::cout << "总算法运行时间: " << execution_time.count() << " 秒" << std::endl;
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
    {
        if (VertexClassify[vh->idx()] == 1)
        {
            mesh1.add_vertex(mesh.point(*vh));
        }
        else if (VertexClassify[vh->idx()] == 2)
        {
            mesh2.add_vertex(mesh.point(*vh));
        }
        else if (VertexClassify[vh->idx()] == 3)
        {
            mesh3.add_vertex(mesh.point(*vh));
        }
    }
    try
    {
        if (!IO::write_mesh(mesh1, "F:/c++ workplace/MeshDenoising/CV1.obj"))
        {
            std::cerr << "Cannot write mesh to file 'CV1.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
    try
    {
        if (!IO::write_mesh(mesh2, "F:/c++ workplace/MeshDenoising/CV2.obj"))
        {
            std::cerr << "Cannot write mesh to file 'CV2.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
    try
    {
        if (!IO::write_mesh(mesh3, "F:/c++ workplace/MeshDenoising/CV3.obj"))
        {
            std::cerr << "Cannot write mesh to file 'CV2.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
    /*for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
    {
        mesh.set_point(*vh, Point[vh->idx()]);
    }*/
    try
    {
        if (!IO::write_mesh(mesh, "F:/c++ workplace/MeshDenoising/Output.obj"))
        {
            std::cerr << "Cannot write mesh to file 'output.obj'" << std::endl;
            return 1;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
        return 1;
    }

    ////获取全聚类信息
    //std::vector<int> julei_faces1(mesh.n_faces(), 0);
    //std::vector<int> julei_faces2(mesh.n_faces(), 0);
    //std::vector<int> julei_faces3(mesh.n_faces(), 0);
    //std::vector<MyMesh::FaceHandle> julei_faces1_iter; // 存储 FaceHandle
    //std::vector<MyMesh::FaceHandle> julei_faces2_iter;
    //std::vector<MyMesh::FaceHandle> julei_faces3_iter;
    //MyMesh julei_mesh1;
    //MyMesh julei_mesh2;
    //MyMesh julei_mesh3;
    //julei_mesh1.request_face_normals();
    //julei_mesh1.request_vertex_normals();
    //julei_mesh2.request_face_normals();
    //julei_mesh2.request_vertex_normals();
    //julei_mesh3.request_face_normals();
    //julei_mesh3.request_vertex_normals();
    //for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
    //    int v = vh->idx();
    //    if (VertexClassify[vh->idx()] != 0) {
    //        std::vector<Group_Face> groups_faces = FeatureVertexGroups[v].groups_faces;
    //        for (int i = 0; i < groups_faces.size(); i++) {
    //            int face_id = groups_faces[i].face_number;
    //            int cluster_id = groups_faces[i].group_number;
    //            if (julei_faces1[face_id] == 0 && cluster_id == 0) {
    //                julei_faces1[face_id]++;
    //                MyMesh::FaceHandle fh = *groups_faces[i].face;
    //                julei_faces1_iter.push_back(fh);
    //            }
    //            else if (julei_faces2[face_id] == 0 && cluster_id == 1) {
    //                julei_faces2[face_id]++;
    //                MyMesh::FaceHandle fh = *groups_faces[i].face;
    //                julei_faces2_iter.push_back(fh);
    //            }
    //            else if (julei_faces3[face_id] == 0 && cluster_id == 2) {
    //                julei_faces3[face_id]++;
    //                MyMesh::FaceHandle fh = *groups_faces[i].face;
    //                julei_faces3_iter.push_back(fh);
    //            }
    //        }
    //    }
    //}
    //std::unordered_map<MyMesh::Point, MyMesh::VertexHandle, PointHasher> vertex_map1;
    //for (int i = 0; i < julei_faces1_iter.size(); i++) {
    //    MyMesh::FaceHandle fh = julei_faces1_iter[i];
    //    std::vector<MyMesh::VertexHandle> face_vhandles;

    //    // 遍历面片的所有顶点
    //    for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(fh); fv_it != mesh.fv_end(fh); ++fv_it) {
    //        MyMesh::VertexHandle source_vh = mesh.vertex_handle(fv_it->idx());
    //        face_vhandles.push_back(source_vh);
    //    }
    //    add_face_to_julei_mesh(mesh, julei_mesh1, face_vhandles, vertex_map1);
    //}
    //std::unordered_map<MyMesh::Point, MyMesh::VertexHandle, PointHasher> vertex_map2;
    //for (int i = 0; i < julei_faces2_iter.size(); i++) {
    //    MyMesh::FaceHandle fh = julei_faces2_iter[i];
    //    std::vector<MyMesh::VertexHandle> face_vhandles;

    //    // 遍历面片的所有顶点
    //    for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(fh); fv_it != mesh.fv_end(fh); ++fv_it) {
    //        MyMesh::VertexHandle source_vh = mesh.vertex_handle(fv_it->idx());
    //        face_vhandles.push_back(source_vh);
    //    }
    //    add_face_to_julei_mesh(mesh, julei_mesh2, face_vhandles, vertex_map2);
    //}
    //std::unordered_map<MyMesh::Point, MyMesh::VertexHandle, PointHasher> vertex_map3;
    //for (int i = 0; i < julei_faces3_iter.size(); i++) {
    //    MyMesh::FaceHandle fh = julei_faces3_iter[i];
    //    std::vector<MyMesh::VertexHandle> face_vhandles;

    //    // 遍历面片的所有顶点
    //    for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(fh); fv_it != mesh.fv_end(fh); ++fv_it) {
    //        MyMesh::VertexHandle source_vh = mesh.vertex_handle(fv_it->idx());
    //        face_vhandles.push_back(source_vh);
    //    }
    //    add_face_to_julei_mesh(mesh, julei_mesh3, face_vhandles, vertex_map3);
    //}
    //try
    //{
    //    if (!IO::write_mesh(julei_mesh1, "F:/c++ workplace/MeshDenoising/全聚类1号.obj"))
    //    {
    //        std::cerr << "Cannot write mesh to file '全聚类.obj'" << std::endl;
    //    }
    //}
    //catch (std::exception& x)
    //{
    //    std::cerr << x.what() << std::endl;
    //}
    //try
    //{
    //    if (!IO::write_mesh(julei_mesh2, "F:/c++ workplace/MeshDenoising/全聚类2号.obj"))
    //    {
    //        std::cerr << "Cannot write mesh to file '全聚类.obj'" << std::endl;
    //    }
    //}
    //catch (std::exception& x)
    //{
    //    std::cerr << x.what() << std::endl;
    //}
    //try
    //{
    //    if (!IO::write_mesh(julei_mesh3, "F:/c++ workplace/MeshDenoising/全聚类3号.obj"))
    //    {
    //        std::cerr << "Cannot write mesh to file '全聚类.obj'" << std::endl;
    //    }
    //}
    //catch (std::exception& x)
    //{
    //    std::cerr << x.what() << std::endl;
    //}
    //std::cout << "以输出全聚类" << std::endl;

    //while (true)//获取单面片聚类信息
    //{
    //    MyMesh output_mesh1;//存放聚类1的面片和顶点
    //    MyMesh output_mesh2;//存放聚类2的面片和顶点
    //    MyMesh output_mesh3;//存放聚类3的面片和顶点
    //    output_mesh1.request_face_normals();
    //    output_mesh1.request_vertex_normals();
    //    output_mesh2.request_face_normals();
    //    output_mesh2.request_vertex_normals();
    //    output_mesh3.request_face_normals();
    //    output_mesh3.request_vertex_normals();
    //    int num;
    //    std::cout << "请输入想要得到的聚类的特征顶点的索引号：" << std::endl;
    //    std::cin >> num;//特征顶点索引号
    //    int k = FeatureVertexGroups[num].K;//顶点的聚类数量
    //    if (k > 3 || k < 2)//判断是否为特征顶点
    //    {
    //        std::cout << "输入错误，请输入正确的特征顶点索引";
    //        continue;
    //    }
    //    std::vector<Group_Face>groups_faces = FeatureVertexGroups[num].groups_faces;
    //    for (int j = 0; j < k; j++)
    //    {
    //        for (int i = 0; i < groups_faces.size(); i++)
    //        {
    //            if (groups_faces[i].group_number == j)
    //            {
    //                MyMesh::FaceIter fh_it = groups_faces[i].face;
    //                std::vector<MyMesh::VertexHandle>  face_vhandles;
    //                for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(*fh_it); fv_it != mesh.fv_end(*fh_it); ++fv_it)
    //                {
    //                    face_vhandles.push_back(mesh.vertex_handle(fv_it->idx()));
    //                }
    //                if (j == 0)
    //                {
    //                    std::vector<MyMesh::VertexHandle> face_vhandles1;
    //                    for (MyMesh::VertexHandle vh : face_vhandles)
    //                    {
    //                        MyMesh::VertexHandle v0 = output_mesh1.add_vertex(mesh.point(vh));
    //                        face_vhandles1.push_back(v0);
    //                    }
    //                    output_mesh1.add_face(face_vhandles1);
    //                }
    //                else if (j == 1)
    //                {
    //                    std::vector<MyMesh::VertexHandle> face_vhandles1;
    //                    for (MyMesh::VertexHandle vh : face_vhandles)
    //                    {
    //                        MyMesh::VertexHandle v0 = output_mesh2.add_vertex(mesh.point(vh));
    //                        face_vhandles1.push_back(v0);
    //                    }
    //                    output_mesh2.add_face(face_vhandles1);
    //                }
    //                else
    //                {
    //                    std::vector<MyMesh::VertexHandle> face_vhandles1;
    //                    for (MyMesh::VertexHandle vh : face_vhandles)
    //                    {
    //                        MyMesh::VertexHandle v0 = output_mesh3.add_vertex(mesh.point(vh));
    //                        face_vhandles1.push_back(v0);
    //                    }
    //                    output_mesh3.add_face(face_vhandles1);
    //                }
    //                
    //            }             
    //        }
    //    }
    //    try
    //    {
    //        if (!IO::write_mesh(output_mesh1, "F:/c++ workplace/MeshDenoising/1号聚类.obj"))
    //        {
    //            std::cerr << "Cannot write mesh to file '1号聚类.obj'" << std::endl;
    //        }
    //    }
    //    catch (std::exception& x)
    //    {
    //        std::cerr << x.what() << std::endl;
    //    }
    //    try
    //    {
    //        if (!IO::write_mesh(output_mesh2, "F:/c++ workplace/MeshDenoising/2号聚类.obj"))
    //        {
    //            std::cerr << "Cannot write mesh to file '2号聚类.obj'" << std::endl;
    //        }
    //    }
    //    catch (std::exception& x)
    //    {
    //        std::cerr << x.what() << std::endl;
    //    }
    //    if (k == 3)
    //    {
    //        try
    //        {
    //            if (!IO::write_mesh(output_mesh3, "F:/c++ workplace/MeshDenoising/3号聚类.obj"))
    //            {
    //                std::cerr << "Cannot write mesh to file '3号聚类.obj'" << std::endl;
    //            }
    //        }
    //        catch (std::exception& x)
    //        {
    //            std::cerr << x.what() << std::endl;
    //        }
    //    }
    //    

    //}

    return 0;
}
