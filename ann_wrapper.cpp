
#include <ANN/ANN.h>
#include <cmath>

struct ANNIndex {
    ANNpointArray points;
    ANNkd_tree* tree;
    int n;
    int dim;
};

extern "C" {

__declspec(dllexport)
void* build_ann_index(const double* data, int n, int dim) {
    if (!data || n <= 0 || dim <= 0) return nullptr;

    ANNIndex* index = new ANNIndex;
    index->n = n;
    index->dim = dim;
    index->points = annAllocPts(n, dim);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            index->points[i][j] = data[i * dim + j];
        }
    }

    index->tree = new ANNkd_tree(index->points, n, dim);

    return index;
}

__declspec(dllexport)
void knn_search(
    void* handle,
    const double* query,
    int k,
    double eps,
    int* indices,
    double* distances
) {
    ANNIndex* index = static_cast<ANNIndex*>(handle);

    index->tree->annkSearch(
        const_cast<double*>(query),
        k,
        indices,
        distances,
        eps
    );

    for (int i = 0; i < k; i++) {
        distances[i] = std::sqrt(distances[i]);
    }
}

__declspec(dllexport)
int radius_search(
    void* handle,
    const double* query,
    double radius,
    double eps,
    int max_results,
    int* indices,
    double* distances
) {
    ANNIndex* index = static_cast<ANNIndex*>(handle);

    int count = index->tree->annkFRSearch(
        const_cast<double*>(query),
        radius * radius,
        max_results,
        indices,
        distances,
        eps
    );

    int returned = (count < max_results) ? count : max_results;

    for (int i = 0; i < returned; i++) {
        distances[i] = std::sqrt(distances[i]);
    }

    return count;
}

__declspec(dllexport)
void free_ann_index(void* handle) {
    if (!handle) return;

    ANNIndex* index = static_cast<ANNIndex*>(handle);

    delete index->tree;
    annDeallocPts(index->points);
    delete index;
}

}
