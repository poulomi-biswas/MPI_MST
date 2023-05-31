
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <mpi.h>

#define ROOT_PROCESS 0

struct Edges { 
    int source, destination;
    double weight;

    Edges() : source(0), destination(0), weight(0.0) {}
    Edges(int s, int d, double w)
        : source(s), destination(d), weight(w) {}
};

std::vector<Edges> readGraph(const std::string& filename) {
    std::vector<Edges> e;

    std::ifstream file(filename);
    if (!file) {
        std::cerr << "failed to open file " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int source, destination;
    while (file >> source >> destination) {
        e.emplace_back(source, destination, 1.0);
    }
    return e;
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#')
            continue;

        std::istringstream iss(line);
        int source, destination;
        double weight;
        if (!(iss >> source >> destination >> weight))
            continue;

        e.emplace_back(source, destination, weight);
    }
    return e;
}

std::vector<Edges> findMST(const std::vector<Edges>& e, int numVertices, int numProcesses, int rank) {
    std::vector<Edges> mst;
    std::vector<bool> visited(numVertices, false);
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> pq;
    if (rank == ROOT_PROCESS) {
        visited[0] = true;
        for (const auto& edge : e) {
            if (edge.source == 0) {
                pq.push(std::make_pair(edge.weight, edge.destination));
            }
        }
    }

    while (!pq.empty()) {
        double weight = pq.top().first;
        int destination = pq.top().second;
        pq.pop();

        if (!visited[destination]) {
            visited[destination] = true;
            mst.emplace_back(destination, destination, weight);

            if (rank == ROOT_PROCESS) {
                for (const auto& edge : e) {
                    if (edge.source == destination && !visited[edge.destination]) {
                        pq.push(std::make_pair(edge.weight, edge.destination));
                    }
                }
            }
        }
    }

    return mst;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == ROOT_PROCESS) {
            std::cerr << "Missing file" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    std::string filename = argv[1];
    std::vector<Edges> e = readGraph(filename);
    int numVertices = e.back().source + 1;
    MPI_Bcast(&numVertices, 1, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);
    std::vector<Edges> localMST = findMST(e, numVertices, numProcesses, rank);
    std::vector<Edges> mst;
    if (rank == ROOT_PROCESS) {
        mst.resize(numVertices - 1);
    }

    MPI_Gather(localMST.data(), localMST.size() * sizeof(Edges), MPI_BYTE,
        mst.data(), localMST.size() * sizeof(Edges), MPI_BYTE,
        ROOT_PROCESS, MPI_COMM_WORLD);
    if (rank == ROOT_PROCESS) {
        std::cout << "Minimum Spanning Tree of the following:" << std::endl;
        for (const auto& edge : mst) {
            std::cout << edge.source << " - " << edge.destination << " : " << edge.weight << std::endl;
        }
    }
    MPI_Finalize();
    return 0;

