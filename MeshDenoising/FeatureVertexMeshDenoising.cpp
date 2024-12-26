#include "FeatureVertexMeshDenoising.h"

void tensorVoting(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal, std::vector<double>& FaceArea,
    std::vector<MyMesh::Point>& FaceCenter, std::vector<int>& VertexClassify, double k, std::vector<double>& VertexS, 
    std::vector<Vector3d>& VertexEigen) {
    // 遍历所有顶点
    MyMesh mesh1;
    MyMesh mesh2;
    MyMesh mesh3;
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
        int vh_id = vh->idx();
        MyMesh::Point x_i = mesh.point(*vh);
        double areaMax = 0;
        Matrix3d matrix;
        matrix <<
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0;
        for (MyMesh::VertexFaceIter vf = mesh.vf_iter(*vh); vf.is_valid(); ++vf)
        {
            int f_id = vf->idx();
            if (FaceArea[f_id] > areaMax)
            {
                areaMax = FaceArea[f_id];
            }
        }
        for (MyMesh::VertexFaceIter vf = mesh.vf_iter(*vh); vf.is_valid(); ++vf)
        {
            int f_id = vf->idx();
            double delta_center = (FaceCenter[f_id] - x_i).norm();
            double weight = FaceArea[f_id] / areaMax * exp(-(delta_center * 3));
            Eigen::Vector3d eigen_normal(FaceNormal[f_id][0], FaceNormal[f_id][1], FaceNormal[f_id][2]);
            Eigen::Matrix3d result = eigen_normal * eigen_normal.transpose();
            matrix += weight * result;
        }
        Eigen::SelfAdjointEigenSolver<Matrix3d> eigensolver(matrix);
        if (eigensolver.info() != Eigen::Success) {
            std::cerr << "Error: Eigen decomposition failed." << std::endl;
        }
        Vector3d eigenvalues = eigensolver.eigenvalues();//获取特征值
        Matrix3d eigenvectors = eigensolver.eigenvectors();//获取特征向量
        std::vector<double> eigenvalue = { eigenvalues(0), eigenvalues(1), eigenvalues(2) };
        std::vector<Vector3d> eigenvector = { eigenvectors.col(0).transpose(),eigenvectors.col(1).transpose(),eigenvectors.col(2).transpose() };
        
        //通过排序，保证eigenvalues(0)>eigenvalues(1)>eigenvalues(2),同时调整对应特征向量
        if (eigenvalue[0] < eigenvalue[1]) std::swap(eigenvalue[0], eigenvalue[1]); std::swap(eigenvector[0], eigenvector[1]);
        if (eigenvalue[0] < eigenvalue[2]) std::swap(eigenvalue[0], eigenvalue[2]); std::swap(eigenvector[0], eigenvector[2]);
        if (eigenvalue[1] < eigenvalue[2]) std::swap(eigenvalue[1], eigenvalue[2]); std::swap(eigenvector[1], eigenvector[2]);

        //if (vh_id == 4074) 
            //std::cout << "1";
        double ratio1 = eigenvalue[0] / eigenvalue[1];
        if (eigenvalue[2] >= k)
        {
            VertexClassify[vh_id] = 2;
            mesh2.add_vertex(x_i);
            VertexS[vh_id] = eigenvalue[2];
            VertexEigen[vh_id] = eigenvector[2];//角点的方向向量随意设置
        }
        else if (eigenvalue[1] >= k)
        {
            if (ratio1 > 3)//圆角边顶点
            {
                VertexClassify[vh_id] = 3;
                mesh3.add_vertex(x_i);
                VertexS[vh_id] = eigenvalue[1] - eigenvalue[2];
                VertexEigen[vh_id] = eigenvector[2];//边点的方向向量设置为特征值最小的特征向量
            }
            else//正常边顶点
            {
                VertexClassify[vh_id] = 1;
                mesh1.add_vertex(x_i);
                VertexS[vh_id] = eigenvalue[1] - eigenvalue[2];
                VertexEigen[vh_id] = eigenvector[2];//边点的方向向量设置为特征值最小的特征向量
            }
        }
        else if (eigenvalue[0] >= k)
        {
            VertexClassify[vh_id] = 0;
            VertexS[vh_id] = eigenvalue[0] - eigenvalue[1];
            VertexEigen[vh_id] = eigenvector[0];//面点的特征向量设置为特征值最大的特征向量
        }
        else
        {
            VertexClassify[vh_id] = 2;
            mesh2.add_vertex(x_i);
            VertexS[vh_id] = eigenvalue[2];
            VertexEigen[vh_id] = eigenvector[2];//角点的特征向量随意设置
        }
        
    }
    try
    {
        if (!IO::write_mesh(mesh1, "F:/c++ workplace/MeshDenoising/TV1.obj"))
        {
            std::cerr << "Cannot write mesh to file 'output.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
    try
    {
        if (!IO::write_mesh(mesh2, "F:/c++ workplace/MeshDenoising/TV2.obj"))
        {
            std::cerr << "Cannot write mesh to file 'output.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
    try
    {
        if (!IO::write_mesh(mesh3, "F:/c++ workplace/MeshDenoising/TV3.obj"))
        {
            std::cerr << "Cannot write mesh to file 'output.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
}

void VertexNormalEstimation(MyMesh& mesh, MyMesh::VertexIter vh, std::vector<MyMesh::Normal>& FaceNormal, 
    std::vector<double>& FaceArea, std::vector<MyMesh::Normal>& VertexNormal)
{
    double AreaAll = 0.0;
    MyMesh::Normal NewN(0, 0, 0);
    for (MyMesh::VertexFaceIter vf = mesh.vf_iter(*vh); vf.is_valid(); ++vf)
    {
        int vf_id = vf->idx();
        AreaAll += FaceArea[vf_id];
    }
    for (MyMesh::VertexFaceIter vf = mesh.vf_iter(*vh); vf.is_valid(); ++vf)
    {
        NewN += FaceNormal[vf->idx()] * FaceArea[vf->idx()] / AreaAll;
    }
    VertexNormal[vh->idx()] = NewN;
    VertexNormal[vh->idx()].normalize();
}

void VertexNormalEstimation(MyMesh& mesh, MyMesh::VertexIter vh, std::vector<MyMesh::Normal>& FaceNormal, 
    std::vector<double>& FaceArea, std::vector<MyMesh::Normal>& VertexNormal, 
    std::vector<Group_Face>& groups_faces, int k, std::vector<Group>& FeatureVertexGroups, 
    std::vector<std::vector<Eigen::Vector4d>>& FitPlane)
{
    std::vector<MyMesh::Normal> groups_normal;
    std::vector<Eigen::Vector4d> groups_plane;
    FeatureVertexGroups[vh->idx()].K = k;
    FeatureVertexGroups[vh->idx()].groups_faces = groups_faces;
    for (int i = 0; i < k; i++)
    {
        std::set<MyMesh::Point> points;
        std::vector<int> FaceNumber;//存储一个聚类所有面片的索引
        for (const auto& fh : groups_faces)
        {
            if (fh.group_number == i)
            {
                FaceNumber.push_back(fh.face_number);
                for (MyMesh::FaceVertexIter fv_it = mesh.fv_begin(*fh.face); fv_it != mesh.fv_end(*fh.face); ++fv_it)
                {
                    points.insert(mesh.point(*fv_it));
                }
                
            }
        }
        int numPoints = points.size();
        Eigen::MatrixXd A(numPoints, 3);
        Eigen::VectorXd b(numPoints);

        // 构建矩阵 A 和向量 b
        int g = 0;
        for (const auto& point : points) {
            A(g, 0) = point[0];
            A(g, 1) = point[1];
            A(g, 2) = point[2];
            g++;
        }
        Eigen::Vector3d centroid = A.colwise().mean();
        Eigen::MatrixXd centered = A.rowwise() - centroid.transpose();
        Eigen::MatrixXd cov = centered.adjoint() * centered / double(points.size() - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
        Eigen::Vector3d normal = eig.eigenvectors().col(0);

        double d = -normal.dot(centroid);

        Eigen::Vector4d plane;
        plane << normal, d;
        MyMesh::Normal facenormal = { normal[0],normal[1],normal[2] };
        facenormal.normalize();
        groups_normal.push_back(facenormal);
        groups_plane.push_back(plane);
    }
    MyMesh::Normal n;
    FeatureVertexGroups[vh->idx()].Nk = groups_normal;
    FitPlane[vh->idx()] = groups_plane;
    for (int i = 0; i < k; i++)
    {
        n += groups_normal[i];
    }
    VertexNormal[vh->idx()] = n / k;
    VertexNormal[vh->idx()].normalize();
    //std::vector<MyMesh::Normal> groups_normal;
    //FeatureVertexGroups[vh->idx()].K = k;
    //FeatureVertexGroups[vh->idx()].groups_faces = groups_faces;
    //for (int i = 0; i < k; i++)
    //{
    //    std::set<MyMesh::Point> points;
    //    std::vector<int> FaceNumber;//存储一个聚类所有面片的索引
    //    for (const auto& fh : groups_faces)
    //    {
    //        if (fh.group_number == i)
    //        {
    //            FaceNumber.push_back(fh.face_number);
    //        }
    //    }
    //    double AreaAll = 0.0;
    //    MyMesh::Normal NewN(0, 0, 0);
    //    for (const auto& fh : FaceNumber)
    //    {
    //        AreaAll += FaceArea[fh];
    //    }
    //    for (const auto& fh : FaceNumber)
    //    {
    //        NewN += FaceNormal[fh] * FaceArea[fh] / AreaAll;
    //    }
    //    groups_normal.push_back(NewN);
    //}
    //MyMesh::Normal n;
    //FeatureVertexGroups[vh->idx()].Nk = groups_normal;
    //for (int i = 0; i < k; i++)
    //{
    //    n += groups_normal[i];
    //}
    //VertexNormal[vh->idx()] = n / k;
    //VertexNormal[vh->idx()].normalize();
}

double cosine_similarity(const Vector3d& v1, const Vector3d& v2) {
    return v1.dot(v2) / (v1.norm() * v2.norm());
}

void k_means_clustering(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal, 
    std::vector<double>& FaceArea, std::vector<MyMesh::Point>& FaceCenter, 
    std::vector<int>& VertexClassify, std::vector<MyMesh::FaceIter>& Face, 
    std::vector<MyMesh::Normal>& VertexNormal, std::vector<Group>& FeatureVertexGroups, 
    int k, MyMesh::VertexIter vh, int vh_id, std::vector<std::vector<Eigen::Vector4d>>& FitPlane) {
    std::vector<Vector3d> center_normals;//存储聚类中心法向量
    std::vector<Group_Face> groups_faces;//存储簇内面片数据
    std::vector<MyMesh::FaceHandle> oneRingFaces;//存储顶点的一环面片
    std::set<int> neighborhood_faces;
    /*for (auto vf_it = mesh.vf_iter(*vh); vf_it.is_valid(); ++vf_it) {
        oneRingFaces.push_back(*vf_it);
    }*/

    for (MyMesh::VertexVertexIter vv_it = mesh.vv_iter(*vh); vv_it.is_valid(); ++vv_it)
    {
        for (MyMesh::VertexFaceIter vvf_it = mesh.vf_iter(*vv_it); vvf_it.is_valid(); ++vvf_it)
        {
            neighborhood_faces.insert(vvf_it->idx());
        }
    }
    for (int face_idx : neighborhood_faces)
    {
        oneRingFaces.push_back(*Face[face_idx]);
    }
    std::priority_queue<FaceComparison, std::vector<FaceComparison>, std::greater<FaceComparison>> maxDiffFaces;

    for (size_t i = 0; i < oneRingFaces.size(); ++i) {
        Group_Face face = { 0, oneRingFaces[i].idx(), Face[oneRingFaces[i].idx()] };
        groups_faces.push_back(face);
        double maxDiff = 0;
        MyMesh::FaceHandle face1 = oneRingFaces[i];
        for (size_t j = 0; j < oneRingFaces.size(); ++j) {
            MyMesh::FaceHandle face2 = oneRingFaces[j];

            // 读取面片法向量
            MyMesh::Normal face1Normal = FaceNormal[face1.idx()];
            MyMesh::Normal face2Normal = FaceNormal[face2.idx()];

            // 计算法向差
            double normalDifference = (face1Normal - face2Normal).length();

            if (maxDiff < normalDifference)
            {
                maxDiff = normalDifference;
            }
        }
        // 将面片加入堆
        maxDiffFaces.push(FaceComparison(face1, maxDiff));
    }
    int q = 0;
    std::vector<MyMesh::FaceHandle>choosed_face;//存储已被选取的面片，防止重复
    while (!maxDiffFaces.empty()) {
        FaceComparison fc = maxDiffFaces.top();
        maxDiffFaces.pop();
        if (q < k)
        {
            MyMesh::Normal normal = FaceNormal[fc.face1.idx()];
            center_normals.push_back(Vector3d(normal[0], normal[1], normal[2]));
            choosed_face.push_back(fc.face1);
            q++;
        }
        else
        {
            break;
        }
    }
    //k-均值聚类
    bool converged = false;
    int clusters[100] = { 0 };
    std::vector<Vector3d> new_center_normals;//存储新聚类中心法向量
    int sum = 0;
    while (!converged && sum < 500) {
        // 将每个面片分配到最近的聚类中心
        for (size_t i = 0; i < oneRingFaces.size(); ++i) {
            int idx = oneRingFaces[i].idx();
            double max_similarity = -1;
            int max_cluster = 0;
            MyMesh::Normal normal = FaceNormal[idx];
            Vector3d normals = Vector3d(normal[0], normal[1], normal[2]);//获取面片法向量
            for (int j = 0; j < k; ++j) {
                double similarity = cosine_similarity(normals, center_normals[j]);//计算余弦相似度
                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    max_cluster = j;
                }
            }
            clusters[i] = max_cluster;
        }

        // 更新聚类中心
        std::vector<Vector3d> center_sum(10);//记录簇中所有法向量的和
        for (size_t i = 0; i < center_sum.size(); ++i)
        {
            center_sum[i].setZero();
        }
        int center_count[100] = { 0 };//记录每个簇的法向量数量
        for (int i = 0; i < oneRingFaces.size(); ++i) {
            int idx = oneRingFaces[i].idx();
            MyMesh::Normal normal = FaceNormal[idx];
            Vector3d normals = Vector3d(normal[0], normal[1], normal[2]);//获取面片法向量
            center_sum[clusters[i]] += normals;
            center_count[clusters[i]] += 1;
        }

        //计算各个聚类的平均法向量，并进行内部比较，选择最接近平均值的面片作为新的聚类中心
        for (int i = 0; i < k; ++i) {
            double max_similarity = -1;
            Vector3d Center_normal;
            Vector3d center_normal = Vector3d(center_sum[i][0] / center_count[i], center_sum[i][1] / center_count[i], center_sum[i][2] / center_count[i]);//获取当前聚类平均法向量
            for (int h = 0; h < oneRingFaces.size(); ++h)
            {
                if (clusters[h] == i)
                {
                    int idx = oneRingFaces[h].idx();
                    MyMesh::Normal normal = FaceNormal[idx];
                    Vector3d normals = Vector3d(normal[0], normal[1], normal[2]);//获取面片法向量
                    double similarity = cosine_similarity(normals, center_normal);//计算余弦相似度
                    if (similarity > max_similarity)
                    {
                        max_similarity = similarity;
                        Center_normal = normals;
                    }
                }
            }
            new_center_normals.push_back(Center_normal);
        }

        // 检查是否收敛（聚类中心是否发生变化）
        converged = true;
        for (int i = 0; i < k; ++i) {
            if ((new_center_normals[i] - center_normals[i]).norm() > 1e-6) {
                converged = false;
                break;
            }
        }
        center_normals = new_center_normals;
        new_center_normals.clear();
        sum++;
    }

    int k_empty = k;
    int a = 10, b = 10;//获取相近簇的索引
    //int kNum[3] = { 0 };
    //int j = 0;
    ////检查每个聚类中心是否有空组，如果有，则将该空组删除
    //for (const auto& fh : groups_faces)
    //{
    //    for (int i = 0; i < k ; ++i)
    //    {
    //        if (fh.group_number == i)
    //        {
    //            kNum[i]++;
    //        }
    //    }
    //}
    //for (int i = 0; i < k; i++)
    //{
    //    if (kNum[i] == 0)
    //    {
    //        k_empty--;
    //        j = i;
    //    }
    //}
    //if (k_empty < k)
    //{
    //    if (k == 3)
    //    {
    //        if (j == 0)
    //        {
    //            for (size_t i = 0; i < oneRingFaces.size(); ++i)
    //            {
    //                if (groups_faces[i].group_number == 2)
    //                {
    //                    groups_faces[i].group_number = 0;
    //                }
    //            }
    //        }
    //        else if (j == 1)
    //        {
    //            for (size_t i = 0; i < oneRingFaces.size(); ++i)
    //            {
    //                if (groups_faces[i].group_number == 2)
    //                {
    //                    groups_faces[i].group_number = 1;
    //                }
    //            }
    //        }
    //        
    //    }
    //    else
    //    {
    //        std::cout << "k==2?k_empty==1!" << std::endl;
    //    }
    //    k = k_empty;
    //}
    //通过计算聚类中心法向量的余弦相似度，判断该顶点是否被误判。
    for (int i = 0; i < k - 1; ++i)
    {
        for (int j = i + 1; j < k; ++j)
        {

            double similarity = cosine_similarity(center_normals[i], center_normals[j]);
            if (similarity > 0.95)
            {
                a = i;
                b = j;
                k_empty--;
            }
        }
    }
    for (size_t i = 0; i < oneRingFaces.size(); ++i)
    {
        groups_faces[i].group_number = clusters[i];
        if (clusters[i] == b)//如果发生误判，将两个相近聚类中心合并
        {
            groups_faces[i].group_number = a;
        }
        if (b == 1 && k > 2)//解决特殊情况，例如遇到1组被0组合并，此时有0，2两组，为防止后续面法线细化接收不到2组，将2组变为1组。
        {
            if (clusters[i] == 2)
            {
                groups_faces[i].group_number = 1;
            }
        }
    }

    if (k_empty < k)
    {
        k = k_empty;
        if (k == 1)
        {
            VertexNormalEstimation(mesh, vh, FaceNormal, FaceArea, VertexNormal);
            VertexClassify[vh_id] = 0;
        }
        else if (k == 2)
        {
            VertexNormalEstimation(mesh, vh, FaceNormal, FaceArea, VertexNormal, groups_faces, k, FeatureVertexGroups, FitPlane);
            VertexClassify[vh_id] = 1;
        }
    }
    else
    {
        VertexNormalEstimation(mesh, vh, FaceNormal, FaceArea, VertexNormal, groups_faces, k, FeatureVertexGroups, FitPlane);
    }
}

void clusterVertex(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal, std::vector<double>& FaceArea, 
    std::vector<MyMesh::Point>& FaceCenter, std::vector<int>& VertexClassify, 
    std::vector<MyMesh::FaceIter>& Face, std::vector<MyMesh::Normal>& VertexNormal, 
    std::vector<Group>& FeatureVertexGroups, std::vector<std::vector<Eigen::Vector4d>>& FitPlane)
{
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
    {
        int vh_id = vh->idx();
        if (VertexClassify[vh_id] == 2)//如果是角顶点
        {
            int k = 3;
            k_means_clustering(mesh, FaceNormal, FaceArea, FaceCenter, VertexClassify, Face, VertexNormal, FeatureVertexGroups, k, vh, vh_id, FitPlane);
        }
        else if (VertexClassify[vh_id] == 1)//如果是边顶点
        {
            int k = 2;
            k_means_clustering(mesh, FaceNormal, FaceArea, FaceCenter, VertexClassify, Face, VertexNormal, FeatureVertexGroups, k, vh, vh_id, FitPlane);
        }
        else
        {
            VertexNormalEstimation(mesh, vh, FaceNormal, FaceArea, VertexNormal);
        }
    }
}

double completeAverageS(MyMesh& mesh, std::vector<double>& FaceArea)
{
    double AvS;
    double All = 0;
    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
    {
        int fid = f_it->idx();
        All += FaceArea[fid];
    }
    AvS = All / FaceArea.size();
    return AvS;
}


// 全局锁，用于保护共享数据
std::mutex coeff_mutex;
std::mutex weight_mutex;
std::mutex A_mutex;
std::mutex matrix_mutex;
std::mutex right_term2_mutex;

void computeElMatrixBlock(int start, int end, MyMesh& mesh, const std::vector<int>& VertexClassify,
    const std::vector<Group>& FeatureVertexGroups, const std::vector<double>& FaceArea,
    double Avs, Eigen::SparseMatrix<double>& Ak_matrix_local, Eigen::SparseMatrix<double>& El_matrix_local) {
    for (MyMesh::VertexIter vh = mesh.vertices_sbegin() + start; vh != mesh.vertices_sbegin() + end; ++vh) {
        int v_id = vh->idx();
        if (VertexClassify[v_id] == 1 || VertexClassify[v_id] == 2) { // 属于特征顶点
            std::vector<Group_Face> groups_faces = FeatureVertexGroups[v_id].groups_faces;
            for (int j = 0; j < FeatureVertexGroups[v_id].K; j++) {
                double GA = 0;
                int k = 0;
                for (const auto& gf : groups_faces) {
                    if (gf.group_number == j) {
                        GA += FaceArea[gf.face_number];
                        k++;
                    }
                }
                if (k == 0) {
                    std::cerr << "Error: No faces in group " << j << " for vertex " << v_id << std::endl;
                    continue;
                }

                double Ak = GA / k / Avs; // 簇K的平均三角形面积与整个网格的平均三角形面积的比率
                double Wk = (k - 1) / k / k;

                // 对于属于同一group的face，更新局部 Ak_matrix_local 和 El_matrix_local
                for (const auto& gf : groups_faces) {
                    if (gf.group_number == j) {
                        int face_id = gf.face_number;

                        // 更新 Ak_matrix_local 行的每个元素
                        for (Eigen::SparseMatrix<double>::InnerIterator it(Ak_matrix_local, face_id); it; ++it) {
                            it.valueRef() *= Ak;
                        }

                        // 更新 El_matrix_local 的对角元素
                        El_matrix_local.coeffRef(face_id, face_id) += Wk;
                    }
                }
            }
        }
    }
}

void computeWeightMatrixBlock(int block_start, int block_end, MyMesh& mesh,
    const std::vector<MyMesh::Normal>& Initial_Normal, const std::vector<MyMesh::Point>& FaceCenter,
    const std::vector<double>& FaceArea, const std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor,
    double SigmaCenter, double SigmaNormal, double Avs,
    std::vector<Eigen::Triplet<double>>& coeff_triple,
    std::vector<Eigen::Triplet<double>>& weight_triple,
    std::vector<Eigen::Triplet<double>>& A_triple)
{
    // 线程局部数据
    std::vector<Eigen::Triplet<double>> coeff_triple_local;
    std::vector<Eigen::Triplet<double>> weight_triple_local;
    std::vector<Eigen::Triplet<double>> A_triple_local;

    for (int i = block_start; i < block_end; ++i) {
        int index_i = i;
        MyMesh::Normal ni = Initial_Normal[index_i];
        MyMesh::Point ci = FaceCenter[index_i];
        std::vector<MyMesh::FaceHandle> face_neighbor = all_face_neighbor[index_i];
        double weight_sum = 0.0;
        double A = FaceArea[index_i] / Avs;
        A_triple_local.push_back(Eigen::Triplet<double>(index_i, index_i, A));
        for (int j = 0; j < (int)face_neighbor.size(); j++)
        {
            int index_j = face_neighbor[j].idx();
            MyMesh::Normal nj = Initial_Normal[index_j];
            MyMesh::Point cj = FaceCenter[index_j];
            double spatial_distance = (ci - cj).length();
            double spatial_weight = std::exp(-0.5 * spatial_distance * spatial_distance / (SigmaCenter * SigmaCenter));
            double range_distance = (ni - nj).length();
            double range_weight = std::exp(-0.5 * range_distance * range_distance / (SigmaNormal * SigmaNormal));
            double weight = FaceArea[index_j] * spatial_weight * range_weight;
            coeff_triple_local.push_back(Eigen::Triplet<double>(index_i, index_j, weight));
            weight_sum += weight;
        }
        if (weight_sum)
        {
            weight_triple_local.push_back(Eigen::Triplet<double>(index_i, index_i, 1.0 / weight_sum));
        }
        else
        {
            std::cerr << "Warning: weight_sum is zero for face " << index_i << std::endl;
        }
        
    }

    // 使用锁保护合并到全局数据结构
    {
        std::lock_guard<std::mutex> coeff_lock(A_mutex);
        A_triple.insert(A_triple.end(), A_triple_local.begin(), A_triple_local.end());
    }

    {
        std::lock_guard<std::mutex> coeff_lock(coeff_mutex);
        coeff_triple.insert(coeff_triple.end(), coeff_triple_local.begin(), coeff_triple_local.end());
    }

    {
        std::lock_guard<std::mutex> weight_lock(weight_mutex);
        weight_triple.insert(weight_triple.end(), weight_triple_local.begin(), weight_triple_local.end());
    }
}

void computeRightTerm1Block(int start, int end, const std::vector<MyMesh::Normal>& Initial_Normal, Eigen::MatrixXd& right_term1) {
    for (int i = start; i < end; ++i) {
        right_term1(i, 0) = Initial_Normal[i][0];
        right_term1(i, 1) = Initial_Normal[i][1];
        right_term1(i, 2) = Initial_Normal[i][2];
    }
}

void computeRightTerm2Block(int start, int end, const std::vector<Group>& FeatureVertexGroups, const std::vector<int>& VertexClassify,
    Eigen::MatrixXd& right_term2) {
    for (int i = start; i < end; ++i) {
        int v_id = i;
        if (VertexClassify[v_id] != 0 && VertexClassify[v_id] != 3) {
            std::vector<Group_Face> groups_faces = FeatureVertexGroups[v_id].groups_faces;
            std::vector<int> K(FeatureVertexGroups[v_id].K, 0);
            for (const auto& gf : groups_faces) {
                K[gf.group_number]++;
            }
            for (const auto& gf : groups_faces) {
                int group_number = gf.group_number;
                right_term2(gf.face_number, 0) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][0];
                right_term2(gf.face_number, 1) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][1];
                right_term2(gf.face_number, 2) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][2];
            }
        }
    }
}

void FacetNormalFieldRefinement(MyMesh& mesh, double smoothness1, double smoothness2,
    double SigmaCenter, double SigmaNormal, std::vector<MyMesh::Normal>& Initial_Normal,
    std::vector<MyMesh::Normal>& FaceNormal_xihua, std::vector<double>& FaceArea,
    std::vector<MyMesh::Point>& FaceCenter, std::vector<int>& VertexClassify,
    std::vector<MyMesh::FaceIter>& Face, std::vector<MyMesh::Normal>& VertexNormal,
    std::vector<Group>& FeatureVertexGroups, std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor)
{
    int n_faces = static_cast<int>(mesh.n_faces());
    int n_vertices = static_cast<int>(mesh.n_vertices());
    Eigen::SparseMatrix<double> weight_matrix(n_faces, n_faces);
    Eigen::SparseMatrix<double> normalize_matrix(n_faces, n_faces);
    Eigen::SparseMatrix<double> A_matrix(n_faces, n_faces);
    Eigen::SparseMatrix<double> Ak_matrix(n_faces, n_faces);
    Eigen::SparseMatrix<double> El_matrix(n_faces, n_faces);
    Eigen::SparseMatrix<double> identity_matrix(n_faces, n_faces);
    Ak_matrix.setIdentity();
    identity_matrix.setIdentity();

    double Avs = completeAverageS(mesh, FaceArea); // 计算所有三角面的平均面积

    std::vector<Eigen::Triplet<double>> coeff_triple;
    std::vector<Eigen::Triplet<double>> weight_triple;
    std::vector<Eigen::Triplet<double>> El_triple;
    std::vector<Eigen::Triplet<double>> A_triple;
    auto startEL = std::chrono::high_resolution_clock::now();  // 记录结束时间

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    // 将计算分成多个块

    // 计算El矩阵
     // 每个线程持有局部的 Ak_matrix_local 和 El_matrix_local
    std::vector<Eigen::SparseMatrix<double>> Ak_matrices_local(num_threads, Eigen::SparseMatrix<double>(n_faces, n_faces));
    std::vector<Eigen::SparseMatrix<double>> El_matrices_local(num_threads, Eigen::SparseMatrix<double>(n_faces, n_faces));

    // 将顶点分块，分配给不同的线程
    int block_size = (n_vertices + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int block_start = i * block_size;
        int block_end = std::min(block_start + block_size, n_vertices);

        threads.push_back(std::thread(computeElMatrixBlock, block_start, block_end, std::ref(mesh),
            std::ref(VertexClassify), std::ref(FeatureVertexGroups), std::ref(FaceArea),
            Avs, std::ref(Ak_matrices_local[i]), std::ref(El_matrices_local[i])));
    }

    // 等待所有线程完成计算
    for (auto& th : threads) {
        th.join();
    }
    threads.clear();
    // 合并局部矩阵到全局矩阵
    for (int i = 0; i < num_threads; ++i) {
        std::lock_guard<std::mutex> lock(matrix_mutex);  // 加锁确保合并时的线程安全

        Ak_matrix += Ak_matrices_local[i];
        El_matrix += El_matrices_local[i];
    }
    auto endEL = std::chrono::high_resolution_clock::now();  // 记录结束时间
    std::chrono::duration<double> execution_time = endEL - startEL;  // 计算运行时间
    std::cout << "EL矩阵计算时间: " << execution_time.count() << " 秒" << std::endl;
    // 分块计算权重矩阵
    
    for (int i = 0; i < num_threads; ++i) {
        int block_start = i * (n_faces / num_threads);
        int block_end = (i == num_threads - 1) ? n_faces : (i + 1) * (n_faces / num_threads);

        threads.push_back(std::thread(computeWeightMatrixBlock, block_start, block_end, std::ref(mesh),
            std::ref(Initial_Normal), std::ref(FaceCenter), std::ref(FaceArea), std::ref(all_face_neighbor),
            SigmaCenter, SigmaNormal, Avs, std::ref(coeff_triple), std::ref(weight_triple), std::ref(A_triple)));
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
    threads.clear();
    auto endESED = std::chrono::high_resolution_clock::now();  // 记录结束时间
    execution_time = endESED - endEL;  // 计算运行时间
    std::cout << "ES和ED矩阵计算时间: " << execution_time.count() << " 秒" << std::endl;

    // 构建权重矩阵
    weight_matrix.setFromTriplets(coeff_triple.begin(), coeff_triple.end());
    normalize_matrix.setFromTriplets(weight_triple.begin(), weight_triple.end());
    A_matrix.setFromTriplets(A_triple.begin(), A_triple.end());

    Eigen::SparseMatrix<double> matrix = identity_matrix - normalize_matrix * weight_matrix;
    Eigen::SparseMatrix<double> coeff_matrix = (1 - smoothness1 - smoothness2) * A_matrix * matrix.transpose() * matrix
        + smoothness1 * A_matrix * identity_matrix
        + smoothness2 * El_matrix * Ak_matrix;

    auto endML = std::chrono::high_resolution_clock::now();  // 记录结束时间
    execution_time = endML - endESED;  // 计算运行时间
    std::cout << "权重矩阵构建时间: " << execution_time.count() << " 秒" << std::endl;

    // 构建右侧项
    Eigen::MatrixXd right_term(mesh.n_faces(), 3);
    Eigen::MatrixXd right_term1(mesh.n_faces(), 3);
    Eigen::MatrixXd right_term2(mesh.n_faces(), 3);
    right_term1.setZero();
    right_term2.setZero();
    
    for (int i = 0; i < num_threads; ++i) {
        int block_start = i * (n_faces / num_threads);
        int block_end = (i == num_threads - 1) ? n_faces : (i + 1) * (n_faces / num_threads);

        threads.push_back(std::thread(computeRightTerm1Block, block_start, block_end, std::ref(Initial_Normal), std::ref(right_term1)));
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }

    // 清空线程容器，准备计算 right_term2
    threads.clear();

    // 多线程计算 right_term2
    for (int i = 0; i < num_threads; ++i) {
        int block_start = i * (n_vertices / num_threads);
        int block_end = (i == num_threads - 1) ? n_vertices : (i + 1) * (n_vertices / num_threads);

        threads.push_back(std::thread(computeRightTerm2Block, block_start, block_end, std::ref(FeatureVertexGroups), std::ref(VertexClassify),
            std::ref(right_term2)));
    }

    // 等待所有线程完成
    for (auto& th : threads) {
        th.join();
    }
    threads.clear();
    right_term = smoothness1 * A_matrix * right_term1 + smoothness2 * Ak_matrix * right_term2;
    
    auto endMR = std::chrono::high_resolution_clock::now();  // 记录结束时间
    execution_time = endMR - endML;  // 计算运行时间
    std::cout << "右侧项构建时间: " << execution_time.count() << " 秒" << std::endl;

    // 计算 Ax = b
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    solver.compute(coeff_matrix);  // 初始化分解
    if (solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return;
    }
    solver.setMaxIterations(1000);  // 最大迭代次数
    solver.setTolerance(1e-6);      // 精度（误差容忍度）
    Eigen::MatrixX3d filtered_normals_matrix = solver.solve(right_term);  // 求解
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solve failed" << std::endl;
    }   
    auto end = std::chrono::high_resolution_clock::now();  // 记录结束时间
    execution_time = end - endMR;  // 计算运行时间
    std::cout << "矩阵计算时间: " << execution_time.count() << " 秒" << std::endl;

    filtered_normals_matrix.rowwise().normalize();
    for (int i = 0; i < FaceNormal_xihua.size(); i++)
    {
        FaceNormal_xihua[i][0] = filtered_normals_matrix(i, 0);
        FaceNormal_xihua[i][1] = filtered_normals_matrix(i, 1);
        FaceNormal_xihua[i][2] = filtered_normals_matrix(i, 2);
        //FaceNormal_xihua[i].normalize();
    }
}




//void FacetNormalFieldRefinement(MyMesh& mesh, double smoothness1, double smoothness2, double SigmaCenter,
//    double SigmaNormal, std::vector<MyMesh::Normal>& Initial_Normal, std::vector<MyMesh::Normal>& FaceNormal_xihua,
//    std::vector<double>& FaceArea, std::vector<MyMesh::Point>& FaceCenter,
//    std::vector<int>& VertexClassify, std::vector<MyMesh::FaceIter>& Face,
//    std::vector<MyMesh::Normal>& VertexNormal, std::vector<Group>& FeatureVertexGroups,
//    std::vector<std::vector<MyMesh::FaceHandle> >& all_face_neighbor)
//{
//    int n_faces = static_cast<int>(mesh.n_faces());
//    Eigen::SparseMatrix<double> weight_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> normalize_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> A_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> El_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> Ak_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> identity_matrix(n_faces, n_faces);
//
//    Ak_matrix.setIdentity();
//    identity_matrix.setIdentity();
//
//    double Avs = completeAverageS(mesh, FaceArea); // 计算所有三角面的平均面积
//
//    // 计算El矩阵
//    std::vector<Eigen::Triplet<double>> El_triple;
//    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
//    {
//        int v_id = vh->idx();
//        if (VertexClassify[v_id] == 1 || VertexClassify[v_id] == 2) // 属于特征顶点
//        {
//            std::vector<Group_Face> groups_faces = FeatureVertexGroups[v_id].groups_faces;
//            for (int j = 0; j < FeatureVertexGroups[v_id].K; j++)
//            {
//                double GA = 0;
//                int k = 0;
//                for (const auto& gf : groups_faces)
//                {
//                    if (gf.group_number == j)
//                    {
//                        GA += FaceArea[gf.face_number];
//                        k++;
//                    }
//                }
//                if (k == 0)
//                {
//                    std::cerr << "Error: No faces in group " << j << " for vertex " << v_id << std::endl;
//                    continue;
//                }
//
//                double Ak = GA / k / Avs; // 簇K的平均三角形面积与整个网格的平均三角形面积的比率
//                double Wk = (k - 1) / k / k;
//                for (const auto& gf : groups_faces)
//                {
//                    if (gf.group_number == j)
//                    {
//                        Ak_matrix.row(gf.face_number) *= Ak;
//                        El_triple.push_back(Eigen::Triplet<double>(gf.face_number, gf.face_number, Wk));
//                    }
//                }
//            }
//        }
//    }
//    El_matrix.setFromTriplets(El_triple.begin(), El_triple.end());
//
//    // 计算Ed和Es
//    std::vector<Eigen::Triplet<double>> coeff_triple, weight_triple, A_triple;
//    double weight_sum;
//    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
//    {
//        int index_i = f_it->idx();
//        MyMesh::Normal ni = Initial_Normal[index_i];
//        MyMesh::Point ci = FaceCenter[index_i];
//        std::vector<MyMesh::FaceHandle> face_neighbor = all_face_neighbor[index_i];
//        weight_sum = 0.0;
//        double A = FaceArea[index_i] / Avs;
//        A_triple.push_back(Eigen::Triplet<double>(index_i, index_i, A));
//
//        for (int i = 0; i < (int)face_neighbor.size(); i++)
//        {
//            int index_j = face_neighbor[i].idx();
//            MyMesh::Normal nj = Initial_Normal[index_j];
//            MyMesh::Point cj = FaceCenter[index_j];
//            double spatial_distance = (ci - cj).length();
//            double spatial_weight = std::exp(-0.5 * spatial_distance * spatial_distance / (SigmaCenter * SigmaCenter));
//            double range_distance = (ni - nj).length();
//            double range_weight = std::exp(-0.5 * range_distance * range_distance / (SigmaNormal * SigmaNormal));
//            double weight = FaceArea[index_j] * spatial_weight * range_weight;
//            coeff_triple.push_back(Eigen::Triplet<double>(index_i, index_j, weight));
//            weight_sum += weight;
//        }
//
//        if (weight_sum)
//        {
//            weight_triple.push_back(Eigen::Triplet<double>(index_i, index_i, 1.0 / weight_sum));
//        }
//        else
//        {
//            std::cerr << "Warning: weight_sum is zero for face " << index_i << std::endl;
//        }
//    }
//
//    weight_matrix.setFromTriplets(coeff_triple.begin(), coeff_triple.end());
//    normalize_matrix.setFromTriplets(weight_triple.begin(), weight_triple.end());
//    A_matrix.setFromTriplets(A_triple.begin(), A_triple.end());
//
//    // 构建系数矩阵
//    Eigen::SparseMatrix<double> matrix = identity_matrix - normalize_matrix * weight_matrix;
//    Eigen::SparseMatrix<double> coeff_matrix = (1 - smoothness1 - smoothness2) * A_matrix * matrix.transpose() * matrix
//        + smoothness1 * A_matrix * identity_matrix
//        + smoothness2 * El_matrix * Ak_matrix;
//
//    // 计算右侧项
//    Eigen::MatrixXd right_term(mesh.n_faces(), 3);
//    Eigen::MatrixXd right_term1(mesh.n_faces(), 3);
//    Eigen::MatrixXd right_term2(mesh.n_faces(), 3);
//    right_term1.setZero();
//    right_term2.setZero();
//
//    for (int i = 0; i < Initial_Normal.size(); i++)
//    {
//        right_term1(i, 0) = Initial_Normal[i][0];
//        right_term1(i, 1) = Initial_Normal[i][1];
//        right_term1(i, 2) = Initial_Normal[i][2];
//    }
//
//    // 计算特征顶点的影响
//    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
//    {
//        int v_id = vh->idx();
//        if (VertexClassify[v_id] != 0) // 属于特征顶点
//        {
//            std::vector<Group_Face> groups_faces = FeatureVertexGroups[v_id].groups_faces;
//            std::vector<int> K(FeatureVertexGroups[v_id].K, 0);
//            for (const auto& gf : groups_faces)
//            {
//                K[gf.group_number]++; // 获取每个聚类的面片数量
//            }
//            for (const auto& gf : groups_faces)
//            {
//                int group_number = gf.group_number;
//                right_term2(gf.face_number, 0) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][0];
//                right_term2(gf.face_number, 1) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][1];
//                right_term2(gf.face_number, 2) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][2];
//            }
//        }
//    }
//
//    right_term = smoothness1 * A_matrix * right_term1 + smoothness2 * Ak_matrix * right_term2;
//
//    // 求解线性方程 Ax = b
//    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
//    solver.analyzePattern(coeff_matrix);
//    if (solver.info() != Eigen::Success)
//    {
//        std::cerr << "Error during analyzePattern: " << solver.info() << std::endl;
//        return;
//    }
//    solver.factorize(coeff_matrix);
//    if (solver.info() != Eigen::Success)
//    {
//        std::cerr << "Error during factorize: " << solver.info() << std::endl;
//        return;
//    }
//
//    Eigen::MatrixX3d filtered_normals_matrix = solver.solve(right_term);
//    if (solver.info() != Eigen::Success)
//    {
//        std::cerr << "Error during solve: " << solver.info() << std::endl;
//        return;
//    }
//
//    filtered_normals_matrix.rowwise().normalize();
//    for (int i = 0; i < FaceNormal_xihua.size(); i++)
//    {
//        FaceNormal_xihua[i][0] = filtered_normals_matrix(i, 0);
//        FaceNormal_xihua[i][1] = filtered_normals_matrix(i, 1);
//        FaceNormal_xihua[i][2] = filtered_normals_matrix(i, 2);
//    }
//}


//void FacetNormalFieldRefinement(MyMesh& mesh, double smoothness1, double smoothness2, double SigmaCenter, 
//    double SigmaNormal,std::vector<MyMesh::Normal>& Initial_Normal, std::vector<MyMesh::Normal>& FaceNormal_xihua,
//    std::vector<double>& FaceArea, std::vector<MyMesh::Point>& FaceCenter,
//    std::vector<int>& VertexClassify, std::vector<MyMesh::FaceIter>& Face,
//    std::vector<MyMesh::Normal>& VertexNormal, std::vector<Group>& FeatureVertexGroups, 
//    std::vector<std::vector<MyMesh::FaceHandle> >& all_face_neighbor)
//{
//    std::vector<Eigen::Triplet<double>> coeff_triple;
//    std::vector<Eigen::Triplet<double>> weight_triple;
//    std::vector<Eigen::Triplet<double>> El_triple;
//    std::vector<Eigen::Triplet<double>> A_triple;
//
//    int n_faces = static_cast<int>(mesh.n_faces());
//    Eigen::SparseMatrix<double> weight_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> normalize_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> A_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> Ak_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> El_matrix(n_faces, n_faces);
//    Eigen::SparseMatrix<double> identity_matrix(n_faces, n_faces);
//    Ak_matrix.setIdentity();
//    identity_matrix.setIdentity();
//
//    double Avs = completeAverageS(mesh, FaceArea); // 计算所有三角面的平均面积
//
//    // 计算El矩阵
//    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
//    {
//        int v_id = vh->idx();
//        if (VertexClassify[v_id] == 1 || VertexClassify[v_id] == 2) // 属于特征顶点
//        {
//            std::vector<Group_Face> groups_faces = FeatureVertexGroups[v_id].groups_faces;
//            for (int j = 0; j < FeatureVertexGroups[v_id].K; j++)
//            {
//                double GA = 0;
//                int k = 0;
//                for (const auto& gf : groups_faces)
//                {
//                    if (gf.group_number == j)
//                    {
//                        GA += FaceArea[gf.face_number];
//                        k++;
//                    }
//                }
//                if (k == 0)
//                {
//                    std::cerr << "Error: No faces in group " << j << " for vertex " << v_id << std::endl;
//                    continue;
//                }
//
//                double Ak = GA / k / Avs; // 簇K的平均三角形面积与整个网格的平均三角形面积的比率
//                double Wk = (k - 1) / k / k;
//                for (const auto& gf : groups_faces)
//                {
//                    if (gf.group_number == j)
//                    {
//                        Ak_matrix.row(gf.face_number) *= Ak;
//                        El_matrix.coeffRef(gf.face_number, gf.face_number) += Wk;
//                    }
//                }
//            }
//        }
//    }
//    El_matrix.setFromTriplets(El_triple.begin(), El_triple.end());
//
//    // 计算Ed和Es
//    for (MyMesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); f_it++)
//    {
//        int index_i = f_it->idx();
//        MyMesh::Normal ni = Initial_Normal[index_i];
//        MyMesh::Point ci = FaceCenter[index_i];
//        std::vector<MyMesh::FaceHandle> face_neighbor = all_face_neighbor[index_i];
//        double weight_sum = 0.0;
//        double A = FaceArea[index_i] / Avs;
//        A_triple.push_back(Eigen::Triplet<double>(index_i, index_i, A));
//        for (int i = 0; i < (int)face_neighbor.size(); i++)
//        {
//            int index_j = face_neighbor[i].idx();
//            MyMesh::Normal nj = Initial_Normal[index_j];
//            MyMesh::Point cj = FaceCenter[index_j];
//            double spatial_distance = (ci - cj).length();
//            double spatial_weight = std::exp(-0.5 * spatial_distance * spatial_distance / (SigmaCenter * SigmaCenter));
//            double range_distance = (ni - nj).length();
//            double range_weight = std::exp(-0.5 * range_distance * range_distance / (SigmaNormal * SigmaNormal));
//            double weight = FaceArea[index_j] * spatial_weight * range_weight;
//            coeff_triple.push_back(Eigen::Triplet<double>(index_i, index_j, weight));
//            weight_sum += weight;
//        }
//        if (weight_sum)
//        {
//            weight_triple.push_back(Eigen::Triplet<double>(index_i, index_i, 1.0 / weight_sum));
//        }
//        else
//        {
//            std::cerr << "Warning: weight_sum is zero for face " << index_i << std::endl;
//        }
//    }
//
//    weight_matrix.setFromTriplets(coeff_triple.begin(), coeff_triple.end());
//    normalize_matrix.setFromTriplets(weight_triple.begin(), weight_triple.end());
//    A_matrix.setFromTriplets(A_triple.begin(), A_triple.end());
//
//    Eigen::SparseMatrix<double> matrix = identity_matrix - normalize_matrix * weight_matrix;
//    Eigen::SparseMatrix<double> coeff_matrix = (1 - smoothness1 - smoothness2) * A_matrix * matrix.transpose() * matrix
//        + smoothness1 * A_matrix * identity_matrix
//        + smoothness2 * El_matrix * Ak_matrix;
//
//    // 构建右侧项
//    Eigen::MatrixXd right_term(mesh.n_faces(), 3);
//    Eigen::MatrixXd right_term1(mesh.n_faces(), 3);
//    Eigen::MatrixXd right_term2(mesh.n_faces(), 3);
//    right_term1.setZero();
//    right_term2.setZero();
//
//    for (int i = 0; i < Initial_Normal.size(); i++)
//    {
//        right_term1(i, 0) = Initial_Normal[i][0];
//        right_term1(i, 1) = Initial_Normal[i][1];
//        right_term1(i, 2) = Initial_Normal[i][2];
//    }
//
//    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
//    {
//        int v_id = vh->idx();
//        if (VertexClassify[v_id] != 0) // 属于特征顶点
//        {
//            std::vector<Group_Face> groups_faces = FeatureVertexGroups[v_id].groups_faces;
//            std::vector<int> K(FeatureVertexGroups[v_id].K, 0);
//            for (const auto& gf : groups_faces)
//            {
//                K[gf.group_number]++; // 获取每个聚类的面片数量
//            }
//            for (const auto& gf : groups_faces)
//            {
//                int group_number = gf.group_number;
//                right_term2(gf.face_number, 0) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][0];
//                right_term2(gf.face_number, 1) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][1];
//                right_term2(gf.face_number, 2) += (K[group_number] - 1) / K[group_number] * FeatureVertexGroups[v_id].Nk[group_number][2];
//            }
//        }
//    }
//
//    right_term = smoothness1 * A_matrix * right_term1 + smoothness2 * Ak_matrix * right_term2;
//
//    // 计算 Ax = b
//    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
//    solver.analyzePattern(coeff_matrix);
//    if (solver.info() != Eigen::Success)
//    {
//        std::cerr << "Error during analyzePattern: " << solver.info() << std::endl;
//        return;
//    }
//    solver.factorize(coeff_matrix);
//    if (solver.info() != Eigen::Success)
//    {
//        std::cerr << "Error during factorize: " << solver.info() << std::endl;
//        return;
//    }
//
//    Eigen::MatrixX3d filtered_normals_matrix = solver.solve(right_term);
//    if (solver.info() != Eigen::Success)
//    {
//        std::cerr << "Error during solve: " << solver.info() << std::endl;
//        return;
//    }
//
//    filtered_normals_matrix.rowwise().normalize();
//    for (int i = 0; i < FaceNormal_xihua.size(); i++)
//    {
//        FaceNormal_xihua[i][0] = filtered_normals_matrix(i, 0);
//        FaceNormal_xihua[i][1] = filtered_normals_matrix(i, 1);
//        FaceNormal_xihua[i][2] = filtered_normals_matrix(i, 2);
//        //FaceNormal_xihua[i].normalize();
//    }
//}

//void VertexPositionUpdate(MyMesh& mesh, std::vector<MyMesh::Point>& Point, 
//    std::vector<MyMesh::Point>& FaceCenter, std::vector<MyMesh::Normal>& FaceNormal_xihua, 
//    std::vector<MyMesh::Normal>& VertexNormal, std::vector<int>& VertexClassify, double smooth1,double smooth2)
//{
//        std::vector<Eigen::Triplet<double>> nf_triple;
//        std::vector<Eigen::Triplet<double>> nv_triple;
//        std::vector<Eigen::Triplet<double>> nv_treature_triple;
//
//        Eigen::SparseMatrix<double> nf_matrix((int)mesh.n_vertices(), (int)mesh.n_vertices());
//        Eigen::SparseMatrix<double> nv_matrix((int)mesh.n_vertices(), (int)mesh.n_vertices());
//        Eigen::SparseMatrix<double> nv_treature_matrix((int)mesh.n_vertices(), (int)mesh.n_vertices());
//        Eigen::SparseMatrix<double> A_matrix((int)mesh.n_vertices(), (int)mesh.n_vertices());
//        Eigen::SparseMatrix<double> identity_matrix((int)mesh.n_vertices(), (int)mesh.n_vertices());
//        identity_matrix.setIdentity();
//
//        Eigen::MatrixXd right_term = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
//        Eigen::MatrixXd v_term = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
//        Eigen::MatrixXd v_terature_term = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
//        Eigen::MatrixXd cf_term = Eigen::MatrixXd::Zero(mesh.n_vertices(), 3);
//
//        for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
//        {
//            int vh_id = vh->idx();
//            double nf = 0;
//            double nv = 0;
//            v_term(vh_id, 0) = mesh.point(*vh)[0];
//            v_term(vh_id, 1) = mesh.point(*vh)[1];
//            v_term(vh_id, 2) = mesh.point(*vh)[2];
//            if (VertexClassify[vh_id] != 0)
//            {
//                v_terature_term(vh_id, 0) = mesh.point(*vh)[0];
//                v_terature_term(vh_id, 1) = mesh.point(*vh)[1];
//                v_terature_term(vh_id, 2) = mesh.point(*vh)[2];
//            }
//
//            for (MyMesh::VFIter vf_it = mesh.vf_iter(*vh); vf_it.is_valid(); ++vf_it)
//            {
//                int vf_id = vf_it->idx();
//                //计算nf内积，指代nf的转置乘以nf
//                double nfTnf = FaceNormal_xihua[vf_id][0] * FaceNormal_xihua[vf_id][0] + FaceNormal_xihua[vf_id][1] * FaceNormal_xihua[vf_id][1] + FaceNormal_xihua[vf_id][2] * FaceNormal_xihua[vf_id][2];
//                cf_term(vh_id, 0) += nfTnf * FaceCenter[vf_id][0];
//                cf_term(vh_id, 1) += nfTnf * FaceCenter[vf_id][1];
//                cf_term(vh_id, 2) += nfTnf * FaceCenter[vf_id][2];
//                nf += nfTnf;
//            }
//            nf_triple.push_back(Eigen::Triplet<double>(vh_id, vh_id, nf));
//            //计算nv内积，指代nv的转置乘以nv
//            nv = VertexNormal[vh_id][0] * VertexNormal[vh_id][0] + VertexNormal[vh_id][1] * VertexNormal[vh_id][1] + VertexNormal[vh_id][2] * VertexNormal[vh_id][2];
//            nv_triple.push_back(Eigen::Triplet<double>(vh_id, vh_id, nv));
//            if (VertexClassify[vh_id] != 0)
//            {
//                nv_treature_triple.push_back(Eigen::Triplet<double>(vh_id, vh_id, 1));
//            }
//        }
//
//        std::cout << "nf_triple size: " << nf_triple.size() << std::endl;
//        std::cout << "nv_triple size: " << nv_triple.size() << std::endl;
//        std::cout << "nv_treature_triple size: " << nv_treature_triple.size() << std::endl;
//
//        nf_matrix.setFromTriplets(nf_triple.begin(), nf_triple.end());
//        nv_matrix.setFromTriplets(nv_triple.begin(), nv_triple.end());
//        nv_treature_matrix.setFromTriplets(nv_treature_triple.begin(), nv_treature_triple.end());
//        //A_matrix = nf_matrix + smooth1 * identity_matrix - 2 * smooth1 * nv_matrix + smooth2 * nv_treature_matrix;
//        A_matrix = nf_matrix + smooth1 * identity_matrix - smooth1 * nv_matrix + smooth2 * nv_treature_matrix;
//        //right_term = cf_term + smooth1 * v_term - 2 * smooth1 * nv_matrix * v_term + smooth2 * v_terature_term;
//        right_term = cf_term + smooth1 * v_term - smooth1 * nv_matrix * v_term + smooth2 * v_terature_term;
//        std::cout << "A_matrix non-zeros: " << A_matrix.nonZeros() << std::endl;
//        std::cout << "Right term size: " << right_term.rows() << "x" << right_term.cols() << std::endl;
//
//        // 计算 Ax = b
//        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
//        solver.analyzePattern(A_matrix);
//        solver.factorize(A_matrix);
//        Eigen::MatrixX3d filtered_point_matrix = solver.solve(right_term);
//
//        for (int i = 0; i < Point.size(); i++)
//        {
//            Point[i][0] = filtered_point_matrix(i, 0);
//            Point[i][1] = filtered_point_matrix(i, 1);
//            Point[i][2] = filtered_point_matrix(i, 2);
//        }
//        for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
//        {
//            mesh.set_point(*vh, Point[vh->idx()]);
//        }
//        //getFaceCenterDatas(mesh, FaceCenter);
//}

void VertexPositionUpdate(MyMesh& mesh, std::vector<MyMesh::Point>& Point,
    std::vector<MyMesh::Point>& FaceCenter, std::vector<MyMesh::Normal>& FaceNormal_xihua,
    std::vector<MyMesh::Normal>& VertexNormal, std::vector<int>& VertexClassify, double smooth1, double smooth2)
{
    std::vector<Eigen::Triplet<double>> A_triple;
    Eigen::VectorXd b_vector = Eigen::VectorXd::Zero((int)mesh.n_vertices() * 3);

    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
    {
        int vh_id = vh->idx();
        Eigen::Matrix3d A_local = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b_local = Eigen::Vector3d::Zero();
        Eigen::Vector3d v(mesh.point(*vh)[0], mesh.point(*vh)[1], mesh.point(*vh)[2]);
        Eigen::Vector3d nv(VertexNormal[vh_id][0], VertexNormal[vh_id][1], VertexNormal[vh_id][2]);

        for (MyMesh::VFIter vf_it = mesh.vf_iter(*vh); vf_it.is_valid(); ++vf_it)
        {
            int vf_id = vf_it->idx();
            Eigen::Vector3d nf(FaceNormal_xihua[vf_id][0], FaceNormal_xihua[vf_id][1], FaceNormal_xihua[vf_id][2]);
            Eigen::Vector3d cf(FaceCenter[vf_id][0], FaceCenter[vf_id][1], FaceCenter[vf_id][2]);
            A_local += nf * nf.transpose();
            b_local += nf * nf.transpose() * cf;
        }

        A_local += smooth1 * (Eigen::Matrix3d::Identity() - nv * nv.transpose());
        b_local += smooth1 * (Eigen::Matrix3d::Identity() - nv * nv.transpose()) * v;

        if (VertexClassify[vh_id] != 0)
        {
            A_local += smooth2 * Eigen::Matrix3d::Identity();
            b_local += smooth2 * v;
        }

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                A_triple.push_back(Eigen::Triplet<double>(vh_id * 3 + i, vh_id * 3 + j, A_local(i, j)));
            }
            b_vector[vh_id * 3 + i] = b_local[i];
        }
    }

    Eigen::SparseMatrix<double> A_matrix((int)mesh.n_vertices() * 3, (int)mesh.n_vertices() * 3);
    A_matrix.setFromTriplets(A_triple.begin(), A_triple.end());

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A_matrix);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "Error during analyzePattern: " << solver.info() << std::endl;
        return;
    }
    solver.factorize(A_matrix);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "Error during factorize: " << solver.info() << std::endl;
        return;
    }

    Eigen::VectorXd solution = solver.solve(b_vector);
    if (solver.info() != Eigen::Success)
    {
        std::cerr << "Error during solve: " << solver.info() << std::endl;
        return;
    }

    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh)
    {
        int vh_id = vh->idx();
        Point[vh_id] = MyMesh::Point(solution[vh_id * 3], solution[vh_id * 3 + 1], solution[vh_id * 3 + 2]);
        mesh.set_point(*vh, Point[vh->idx()]);
    }
}

void IsolatedFeatureElimination(MyMesh& mesh, std::vector<int>& VertexClassify) {
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
        int index_i = vh->idx();
        if (VertexClassify[index_i] != 0) {
            std::unordered_set<MyMesh::VertexHandle> neighbors = get_Vertex_N_ring_neighbors(mesh, *vh, 3);
            int u = 1;
            for (const auto& neighbor_vh : neighbors)
            {
                if (VertexClassify[neighbor_vh.idx()] != 0)
                {
                    u = 0;
                    break;
                }
            }
            if (u == 1)
            {
                VertexClassify[index_i] = 0;
            }
        }
    }
}

void PseudoCornerFeatureElimination(MyMesh& mesh, std::vector<int>& VertexClassify, std::vector<double>& VertexS) {
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
        int index_i = vh->idx();
        if (VertexClassify[index_i] == 2) {
            double meanS = VertexS[index_i];
            std::unordered_set<MyMesh::VertexHandle> neighbors = get_Vertex_N_ring_neighbors(mesh, *vh, 1);
            for (const auto& neighbor_vh : neighbors)
            {
                if (VertexClassify[neighbor_vh.idx()] == 2)
                {
                    double neiS = VertexS[neighbor_vh.idx()];
                    if (meanS > neiS)
                    {
                        VertexClassify[neighbor_vh.idx()] = 0;
                    }
                    else
                    {
                        VertexClassify[index_i] = 0;
                        break;
                    }
                }
            }
        }
    }
}

void WeakFeaturePointRecognition(MyMesh& mesh, std::vector<int>& VertexClassify, std::vector<Vector3d>& VertexEigen) {
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
        int vh_idx = vh->idx();
        if (VertexClassify[vh_idx] == 0)
        {
            int n = 0;
            for (MyMesh::VertexVertexIter v_vh = mesh.vv_iter(*vh); v_vh.is_valid(); ++v_vh)
            {
                int v_vh_idx = v_vh->idx();
                if (VertexClassify[v_vh_idx] == 1 || VertexClassify[v_vh_idx] == 3)
                {
                    Vector3d FVE = VertexEigen[v_vh_idx];
                    Vector3d vertex1(mesh.point(*vh)[0], mesh.point(*vh)[1], mesh.point(*vh)[2]);
                    Vector3d vertex2(mesh.point(*v_vh)[0], mesh.point(*v_vh)[1], mesh.point(*v_vh)[2]);
                    Vector3d normalVertex = (vertex1 - vertex2).normalized();
                    double dotProduct = normalVertex.dot(FVE);
                    double magnitude1 = normalVertex.norm();
                    double magnitude2 = FVE.norm();
                    double angle;
                    if (magnitude1 == 0 || magnitude2 == 0)
                    {
                        angle = 0;
                    }
                    else
                    {
                        double cos_theta = dotProduct / (magnitude1 * magnitude2);
                        cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
                        angle = std::acos(cos_theta) * 180 / M_PI;
                    }
                    if (angle > 0 && angle < 30 || angle > 150 && angle < 180) {
                        n++;
                    }
                }
            }
            if (n >= 2) {
                VertexClassify[vh_idx] = 3;
            }

        }
    }
}

void RoundedFeaturePointFiltering(MyMesh& mesh, std::vector<int>& VertexClassify, std::vector<Vector3d>& VertexEigen) {
    for (MyMesh::VertexIter vh = mesh.vertices_begin(); vh != mesh.vertices_end(); ++vh) {
        int vh_idx = vh->idx();
        if (VertexClassify[vh_idx] == 3)
        {
            int n = 0, m = 0;
            for (MyMesh::VertexVertexIter v_vh = mesh.vv_iter(*vh); v_vh.is_valid(); ++v_vh)
            {
                int v_vh_idx = v_vh->idx();
                if (VertexClassify[v_vh_idx] == 3)
                {
                    Vector3d FVE = VertexEigen[v_vh_idx];
                    Vector3d vertex1(mesh.point(*vh)[0], mesh.point(*vh)[1], mesh.point(*vh)[2]);
                    Vector3d vertex2(mesh.point(*v_vh)[0], mesh.point(*v_vh)[1], mesh.point(*v_vh)[2]);
                    Vector3d normalVertex = (vertex1 - vertex2).normalized();
                    double dotProduct = normalVertex.dot(FVE);
                    double magnitude1 = normalVertex.norm();
                    double magnitude2 = FVE.norm();
                    double angle;
                    if (magnitude1 == 0 || magnitude2 == 0)
                    {
                        angle = 0;
                    }
                    else
                    {
                        double cos_theta = dotProduct / (magnitude1 * magnitude2);
                        cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
                        angle = std::acos(cos_theta) * 180 / M_PI;
                    }
                    if (angle > 0 && angle < 30 || angle > 150 && angle < 180) {
                        n++;
                    }
                    else
                    {
                        m++;
                    }
                }
            }
            if (n >= 2 && m == 0 || n < 2 && m == 0) {
                VertexClassify[vh_idx] = 1;
            }

        }
    }
    //查看特征点筛选情况
    MyMesh mesh1;
    MyMesh mesh2;
    MyMesh mesh3;
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
        if (!IO::write_mesh(mesh1, "F:/c++ workplace/MeshDenoising/BV1.obj"))
        {
            std::cerr << "Cannot write mesh to file 'output.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
    try
    {
        if (!IO::write_mesh(mesh2, "F:/c++ workplace/MeshDenoising/BV2.obj"))
        {
            std::cerr << "Cannot write mesh to file 'output.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
    try
    {
        if (!IO::write_mesh(mesh3, "F:/c++ workplace/MeshDenoising/BV3.obj"))
        {
            std::cerr << "Cannot write mesh to file 'output.obj'" << std::endl;
        }
    }
    catch (std::exception& x)
    {
        std::cerr << x.what() << std::endl;
    }
}

std::unordered_set<MyMesh::VertexHandle> get_Vertex_N_ring_neighbors(MyMesh& mesh, MyMesh::VertexHandle vh,int N) {
    std::unordered_set<MyMesh::VertexHandle> visited; // 已访问顶点集合
    std::queue<std::pair<MyMesh::VertexHandle, int>> to_visit; // 待访问顶点队列，包含顶点和当前深度

    // 将起始顶点放入队列
    to_visit.push({ vh, 0 });
    visited.insert(vh);
    while (!to_visit.empty()) {
        auto current = to_visit.front();
        to_visit.pop();
        MyMesh::VertexHandle current_vh = current.first;
        int depth = current.second;
        if (depth < N) { // 仅当深度小于N时继续搜索
            for (auto vv_it = mesh.vv_iter(current_vh); vv_it.is_valid(); ++vv_it) {
                MyMesh::VertexHandle neighbor_vh = *vv_it;
                if (visited.find(neighbor_vh) == visited.end()) {
                    visited.insert(neighbor_vh);
                    to_visit.push({ neighbor_vh, depth + 1 });
                }
            }
        }
    }

    visited.erase(vh); // 去除起始顶点

    return visited;
}


void adjustFeatureVertexPositions(MyMesh& mesh, std::vector<MyMesh::Normal>& filtered_normals,
    std::vector<MyMesh::Point>& Point, int PointNum, bool fixed_boundary, std::vector<int>& VertexClassify,
    std::vector<Group>& FeatureVertexGroups, std::vector<std::vector<Eigen::Vector4d>>& FitPlane)
{
    std::vector<MyMesh::Point> new_points(mesh.n_vertices());

    std::vector<MyMesh::Point> centroid;

    for (int iter = 0; iter < PointNum; iter++)
    {
        getFaceCentroid(mesh, centroid);
        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
        {
            MyMesh::Point p = mesh.point(*v_it);
            MyMesh::Point p1(0.0, 0.0, 0.0);
            MyMesh::Point p2(0.0, 0.0, 0.0);
            if (fixed_boundary && mesh.is_boundary(*v_it) || VertexClassify[v_it->idx()] == 0)
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
                if (face_num != 0) p1 += temp_point / face_num;
                std::vector<MyMesh::Normal> groups_normal;
                std::vector<Eigen::Vector4d> groups_plane;
                groups_normal = FeatureVertexGroups[v_it->idx()].Nk;
                groups_plane = FitPlane[v_it->idx()];
                for (int i = 0; i < groups_normal.size(); i++)
                {
                    Eigen::Vector3d a(groups_normal[i][0], groups_normal[i][1], groups_normal[i][2]);
                    Eigen::Vector3d b(p[0], p[1], p[2]);
                    double dot_product = a.dot(b);
                    double term = dot_product + groups_plane[i][3];
                    p2 -= MyMesh::Point(a[0] * term, a[1] * term, a[2] * term);
                }
                p += 0.9 * p1 + 0.1 * p2;
                new_points.at(v_it->idx()) = p;
            }
        }

        for (MyMesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); v_it++)
            mesh.set_point(*v_it, new_points[v_it->idx()]);
    }
}


