#pragma once

namespace glso {


template <typename T, int D>
class VertexDescriptor {
private:


public:

void update(T* vertex, const T* update) const;

};

// V is the number of vertices
// O is the dimension of the observation
// E is the dimension of the error vector
template <typename T, int V, int O, int E>
class FactorDescriptor {
public:

// T** get_vertices(Graph* graph);
// const get_vertex_types() 

void compute_error(const T** vertices, const T* obs, T* error) const;

};


class Vertex {
    private:

    size_t id;

    public:

    void set_id(size_t id) {
        this->id = id;
    }

};

class Factor {


};

class Graph {

    public:

    void add_vertex_descriptor();

    // void add_vertex(Vertex & vertex) {
        
    // }

};

class Optimizer {


};

}


