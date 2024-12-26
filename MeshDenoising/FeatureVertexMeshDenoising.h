#pragma once
#ifndef FEATUREVERTEXMESHDENOISING_H
#define FEATUREVERTEXMESHDENOISING_H
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <queue>
#include <mutex>
#include <thread>
#include <unordered_set>
#include "MeshDenoisingBase.h"
// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
using namespace OpenMesh;

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Vector3d Vector3d;

struct FaceComparison {
    MyMesh::FaceHandle face1;
    double normalDifference;

    FaceComparison(MyMesh::FaceHandle f1, double diff) : face1(f1), normalDifference(diff) {}

    // �������Ҫ�ıȽϺ���
    bool operator>(const FaceComparison& other) const {
        return normalDifference > other.normalDifference;
    }
};

//���е���Ƭ����
struct Group_Face {
    int group_number; //�����ڶ���ļ�����
    int face_number; //���������е�λ��
    MyMesh::FaceIter face;
};

//���������������
struct Group {
    int K;//�������
    std::vector<Group_Face>groups_faces;//�������������Ƭ
    std::vector<MyMesh::Normal>Nk;//ÿ����Ĵ����Է���
};

void tensorVoting(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal, std::vector<double>& FaceArea, 
	std::vector<MyMesh::Point>& FaceCenter, std::vector<int>& VertexClassify, double k, std::vector<double>& VertexS, 
    std::vector<Vector3d>& VertexEigen);//����ͶƱ�㷨
void VertexNormalEstimation(MyMesh& mesh, MyMesh::VertexIter vh, std::vector<MyMesh::Normal>& FaceNormal, 
	std::vector<double>& FaceArea, std::vector<MyMesh::Normal>& VertexNormal);//���㷨�򳡼��㣨����������棩
void VertexNormalEstimation(MyMesh& mesh, MyMesh::VertexIter vh, std::vector<MyMesh::Normal>& FaceNormal, 
	std::vector<double>& FaceArea, std::vector<MyMesh::Normal>& VertexNormal, 
	std::vector<Group_Face>& groups_faces, int k, std::vector<Group>& FeatureVertexGroups, 
    std::vector<std::vector<Eigen::Vector4d>>& FitPlane);//���㷨�򳡼��㣨��������棩
double cosine_similarity(const Vector3d& v1, const Vector3d& v2);//���������������������ƶ�
void k_means_clustering(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal, 
    std::vector<double>& FaceArea, std::vector<MyMesh::Point>& FaceCenter, 
    std::vector<int>& VertexClassify, std::vector<MyMesh::FaceIter>& Face, 
    std::vector<MyMesh::Normal>& VertexNormal, std::vector<Group>& FeatureVertexGroups, 
    int k, MyMesh::VertexIter vh, int vh_id, std::vector<std::vector<Eigen::Vector4d>>& FitPlane);
void clusterVertex(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal, std::vector<double>& FaceArea, 
    std::vector<MyMesh::Point>& FaceCenter, std::vector<int>& VertexClassify, 
    std::vector<MyMesh::FaceIter>& Face, std::vector<MyMesh::Normal>& VertexNormal, 
    std::vector<Group>& FeatureVertexGroups, std::vector<std::vector<Eigen::Vector4d>>& FitPlane);//�����������
double completeAverageS(MyMesh& mesh, std::vector<double>& FaceArea);//����������������Ƭ��ƽ�����
void FacetNormalFieldRefinement(MyMesh& mesh, double smoothness1, double smoothness2, double SigmaCenter, double SigmaNormal,
    std::vector<MyMesh::Normal>& Initial_Normal, std::vector<MyMesh::Normal>& FaceNormal_xihua,
    std::vector<double>& FaceArea, std::vector<MyMesh::Point>& FaceCenter,
    std::vector<int>& VertexClassify, std::vector<MyMesh::FaceIter>& Face,
    std::vector<MyMesh::Normal>& VertexNormal, std::vector<Group>& FeatureVertexGroups, 
    std::vector<std::vector<MyMesh::FaceHandle> >& all_face_neighbor);//��Ƭ����ϸ��
void VertexPositionUpdate(MyMesh& mesh, std::vector<MyMesh::Point>& Point, 
    std::vector<MyMesh::Point>& FaceCenter, std::vector<MyMesh::Normal>& FaceNormal_xihua, 
    std::vector<MyMesh::Normal>& VertexNormal, std::vector<int>& VertexClassify, double smooth1, double smooth2);//����λ�ø���
void IsolatedFeatureElimination(MyMesh& mesh, std::vector<int>& VertexClassify);//��������������
void PseudoCornerFeatureElimination(MyMesh& mesh, std::vector<int>& VertexClassify, std::vector<double>& VertexS);//α�ǵ�����������
void WeakFeaturePointRecognition(MyMesh& mesh, std::vector<int>& VertexClassify, std::vector<Vector3d>& VertexEigen);//��������ʶ��
void RoundedFeaturePointFiltering(MyMesh& mesh, std::vector<int>& VertexClassify, std::vector<Vector3d>& VertexEigen);//Բ�Ƕ���ɸѡ
std::unordered_set<MyMesh::VertexHandle> get_Vertex_N_ring_neighbors(MyMesh& mesh, MyMesh::VertexHandle vh, int N);//��Ѱ����N�������ڵĶ���
double distancePointToPlane(const Eigen::Vector4d& plane, const Eigen::Vector3d& point);
void adjustFeatureVertexPositions(MyMesh& mesh, std::vector<MyMesh::Normal>& filtered_normals,
    std::vector<MyMesh::Point>& Point, int PointNum, bool fixed_boundary, std::vector<int>& VertexClassify,
    std::vector<Group>& FeatureVertexGroups, std::vector<std::vector<Eigen::Vector4d>>& FitPlane);
void computeElMatrixBlock(int start, int end, MyMesh& mesh, const std::vector<int>& VertexClassify,
    const std::vector<Group>& FeatureVertexGroups, const std::vector<double>& FaceArea,
    double Avs, Eigen::SparseMatrix<double>& Ak_matrix_local, Eigen::SparseMatrix<double>& El_matrix_local);
void computeWeightMatrixBlock(int block_start, int block_end, MyMesh& mesh,
    const std::vector<MyMesh::Normal>& Initial_Normal, const std::vector<MyMesh::Point>& FaceCenter,
    const std::vector<double>& FaceArea, const std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor,
    double SigmaCenter, double SigmaNormal, double Avs,
    std::vector<Eigen::Triplet<double>>& coeff_triple,
    std::vector<Eigen::Triplet<double>>& weight_triple,
    std::vector<Eigen::Triplet<double>>& A_triple);
void computeRightTerm1Block(int start, int end, const std::vector<MyMesh::Normal>& Initial_Normal, Eigen::MatrixXd& right_term1);
void computeRightTerm2Block(int start, int end, const std::vector<Group>& FeatureVertexGroups, const std::vector<int>& VertexClassify,
    Eigen::MatrixXd& right_term2);
#endif