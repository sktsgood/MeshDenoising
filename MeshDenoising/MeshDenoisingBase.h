#pragma once
#ifndef MESHDENOISINGBASE_H
#define MESHDENOISINGBASE_H
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <thread>
#include <mutex>
// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
using namespace OpenMesh;

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

enum FaceNeighborType { kVertexBased, kEdgeBased };//—°‘Ò√Ê¡Ï”Ú∑∂Œß
void getFaceArea(MyMesh& mesh, std::vector<double>& area);
void getFaceCentroid(MyMesh& mesh, std::vector<MyMesh::Point>& centroid);
void getFaceNormal(MyMesh& mesh, std::vector<MyMesh::Normal>& normals);
void getFaceNeighbor(MyMesh& mesh, MyMesh::FaceHandle fh, int ring, FaceNeighborType face_neighbor_type,
	std::vector<MyMesh::FaceHandle>& face_neighbor);
void getFaceIter(MyMesh& mesh, std::vector<MyMesh::FaceIter>& face_iter);
double computeSigmaCenter(MyMesh mesh, std::vector<MyMesh::Point> FaceCenter, double multiple);
void adjustFaceNormals(MyMesh& mesh, std::vector<MyMesh::Normal>& FaceNormal,
	std::vector<MyMesh::Normal>& NewFaceNormal, std::vector<double>& FaceArea,
	std::vector<MyMesh::Point>& FaceCenter, double SigmaCenter, double SigmaNormal,
	std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor);
void adjustVertexPositions(MyMesh& mesh, std::vector<MyMesh::Normal>& filtered_normals, 
	std::vector<MyMesh::Point>& Point, int PointNum, bool fixed_boundary);
void adjustNonFeatureVertexPositions(MyMesh& mesh, std::vector<MyMesh::Normal>& filtered_normals,
	std::vector<MyMesh::Point>& Point, int PointNum, bool fixed_boundary, std::vector<int>& VertexClassify, 
	std::vector<MyMesh::Normal>& VertexNormal);
void computeFaceNormals(MyMesh& mesh, std::vector<MyMesh::Normal>& normals, int start, int end);
void computeAdjustedFaceNormals(MyMesh& mesh, const std::vector<MyMesh::Normal>& FaceNormal,
	std::vector<MyMesh::Normal>& NewFaceNormal, const std::vector<double>& FaceArea,
	const std::vector<MyMesh::Point>& FaceCenter, double SigmaCenter, double SigmaNormal,
	const std::vector<std::vector<MyMesh::FaceHandle>>& all_face_neighbor,
	int start, int end);
#endif