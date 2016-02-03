/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#include <bh.h>
#include <bh_dag.h>
#include <bh_fuse_cache.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/foreach.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/algorithm/string/predicate.hpp> //For iequals()
#include <boost/graph/connected_components.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <regex>
#include <vector>
#include <map>
#include <iterator>
#include <stdio.h>
#include <exception>
#include <omp.h>

using namespace std;
using namespace boost;
using namespace bohrium;
using namespace bohrium::dag;

//FILO Task Queue thread safe
class TaskQueue
{
public:
    typedef pair<vector<bool>, unsigned int> Task;
private:
    mutex mtx;
    condition_variable non_empty;
    vector<Task> tasks;
    unsigned int nwaiting;
    const unsigned int nthreads;
    bool finished;
public:
    TaskQueue(unsigned int nthreads):nwaiting(0), nthreads(nthreads), finished(false){}

    void push(const vector<bool> &mask, unsigned int offset)
    {
        unique_lock<mutex> lock(mtx);
        tasks.push_back(make_pair(mask, offset));
        non_empty.notify_one();
    }

    Task pop()
    {
        unique_lock<mutex> lock(mtx);
        if(++nwaiting >= nthreads and tasks.size() == 0)
        {
            finished = true;
            non_empty.notify_all();
            throw overflow_error("Out of work");
        }

        while(tasks.size() == 0 and not finished)
            non_empty.wait(lock);

        if(finished)
            throw overflow_error("Out of work");

        Task ret = tasks.back();
        tasks.pop_back();
        --nwaiting;
        return ret;
    }
};

/* Help function that fuses the edges in 'edges2explore' where the 'mask' is true */
pair<int64_t,bool> fuse_mask(int64_t best_cost, const vector<EdgeW> &edges2explore,
                             const GraphDW &graph, const vector<bool> &mask, bh_ir &bhir,
                             GraphD &dag)
{
    bool fusibility=true;
    vector<EdgeW> edges2merge;
    unsigned int i=0;
    BOOST_FOREACH(const EdgeW &e, edges2explore)
    {
        if(mask[i++])
            edges2merge.push_back(e);
    }

    //Help function to find the new location
    struct find_new_location
    {
        Vertex operator()(const map<Vertex, Vertex> &loc_map, Vertex v)
        {
            Vertex v_mapped = loc_map.at(v);
            if(v_mapped == v)
                return v;
            else
                return (*this)(loc_map, v_mapped);
        }
    }find_loc;

    //'loc_map' maps a vertex before the merge to the corresponding vertex after the merge
    map<Vertex, Vertex> loc_map;
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        loc_map[v] = v;
    }

    //Lets record the merges into 'loc_map'
    BOOST_FOREACH(const EdgeW &e, edges2merge)
    {
        Vertex v1 = find_loc(loc_map, source(e, graph.bglW()));
        Vertex v2 = find_loc(loc_map, target(e, graph.bglW()));
        loc_map[v1] = v2;
    }

    //Pack 'loc_map' such that all keys maps directly to a new vertex thus after
    //this point there is no need to call find_loc().
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        Vertex v_mapped = find_loc(loc_map, loc_map.at(v));
        if(v_mapped != v)
            loc_map[v] = v_mapped;
    }

    //Create the new vertices and insert instruction topologically
    map<Vertex, bh_ir_kernel> new_vertices;
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        if(loc_map.at(v) == v)
            new_vertices[v] = bh_ir_kernel(bhir);
    }
    vector<Vertex> topological_order;
    topological_sort(dag, back_inserter(topological_order));
    BOOST_REVERSE_FOREACH(Vertex vertex, topological_order)
    {
        Vertex v = loc_map.at(vertex);
        bh_ir_kernel &k = new_vertices.at(v);
        BOOST_FOREACH(uint64_t idx, dag[vertex].instr_indexes())
        {
            if(not k.fusible(idx))
                fusibility = false;
            k.add_instr(idx);
        }
    }

    //TODO: Remove this assert check
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        if(loc_map.at(v) == v)
            assert(new_vertices[v].instr_indexes().size() > 0);
    }

    //Find the total cost
    int64_t cost=0;
    BOOST_FOREACH(const bh_ir_kernel &k, new_vertices | adaptors::map_values)
    {
        cost += k.cost();
    }

    //Check if we need to continue
    if(cost >= best_cost or not fusibility)
        return make_pair(cost,false);

    //Merge the vertice in the DAG
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        Vertex loc_v = loc_map.at(v);
        if(loc_v == v)
        {
            dag[v] = new_vertices.at(v);
            assert(dag[v].instr_indexes().size() > 0);
        }
        else//Lets merge 'v' into 'loc_v'
        {
            BOOST_FOREACH(Vertex a, adjacent_vertices(v, dag))
            {
                a = loc_map.at(a);
                if(a != loc_v)
                    add_edge(loc_v, a, dag);
            }
            BOOST_FOREACH(Vertex a, inv_adjacent_vertices(v, dag))
            {
                a = loc_map.at(a);
                if(a != loc_v)
                    add_edge(a, loc_v, dag);
            }
            clear_vertex(v, dag);
            dag[v] = bh_ir_kernel(bhir);
        }
    }

    //TODO: remove assert check
    BOOST_FOREACH(Vertex v, vertices(dag))
    {
        if(dag[loc_map.at(v)].instr_indexes().size() == 0)
        {
            cout << v << endl;
            cout << loc_map.at(v) << endl;
            assert(1 == 2);
            exit(-1);
        }
    }

    //Check for cycles
    if(cycles(dag))
    {
        return make_pair(cost,false);
    }

    assert(cost == (int64_t)dag_cost(dag));
    return make_pair(cost,true);
}

/* Find the optimal solution through branch and bound */
void branch_n_bound(bh_ir &bhir, GraphDW &dag, const vector<EdgeW> &edges2explore,
                    const vector<bool> &init_mask, unsigned int init_offset=0)
{
    //We use the greedy algorithm to find a good initial guess
    int64_t best_cost;
    GraphD best_dag;
    {
        GraphDW new_dag(dag);
        fuse_greedy(new_dag);
        best_dag = new_dag.bglD();
        best_cost = dag_cost(best_dag);
    }
    uint64_t purge_count=0;
    uint64_t explore_count=0;

    TaskQueue tasks(omp_get_max_threads());
    tasks.push(init_mask, init_offset);
    #pragma omp parallel
    {
    while(1)
    {
        vector<bool> mask;
        unsigned int offset;
        try{
            tie(mask, offset) = tasks.pop();
        }catch(overflow_error &e){
            break;
        }

        //Fuse the task
        GraphD new_dag(dag.bglD());
        bool fusibility;
        int64_t cost;
        tie(cost, fusibility) = fuse_mask(best_cost, edges2explore, dag, mask, bhir, new_dag);

        if(explore_count%10000 == 0)
        {
            #pragma omp critical
            {
                cout << "[" << (double) explore_count << "][";
                BOOST_FOREACH(bool b, mask)
                {
                    if(b){cout << "1";}else{cout << "0";}
                }
                cout << "] search: ";
                cout << (double) explore_count + purge_count << " / " << pow(2.0, (int)mask.size());
                cout << ", purged: " << (double) purge_count << ", best_cost: " << best_cost << endl;
            }
        }
        #pragma omp critical
        ++explore_count;

        if(cost >= best_cost)
        {
            #pragma omp critical
            purge_count += pow(2.0, (int)(mask.size()-offset))-1;
            continue;
        }
        if(fusibility)
        {
            #pragma omp critical
            {
                //Lets save the new best dag
                best_cost = cost;
                best_dag = new_dag;
                assert(dag_validate(best_dag));
                purge_count += pow(2.0, (int)(mask.size()-offset));
            }
            continue;
        }
        //for(unsigned int i=offset; i<mask.size(); ++i) //breadth first
        for(int i=mask.size()-1; i>= (int)offset; --i)   //depth first
        {
            vector<bool> m1(mask);
            m1[i] = false;
            tasks.push(m1, i+1);
        }
    }}
    dag = best_dag;
}

void get_edges2explore(const GraphDW &dag, vector<EdgeW> &edges2explore)
{
    //The list of edges that we should try to merge
    BOOST_FOREACH(const EdgeW &e, edges(dag.bglW()))
    {
        edges2explore.push_back(e);
    }
    sort_weights(dag.bglW(), edges2explore);
    string order;
    {
        const char *t = getenv("BH_FUSER_OPTIMAL_ORDER");
        if(t == NULL)
            order ="regular";
        else
            order = t;
    }
    if(not iequals(order, "regular"))
    {
        if(iequals(order, "reverse"))
        {
            reverse(edges2explore.begin(), edges2explore.end());
        }
        else if(iequals(order, "random"))
        {
            random_shuffle(edges2explore.begin(), edges2explore.end());
        }
        else
        {
            cerr << "FUSER-OPTIMAL: unknown BH_FUSER_OPTIMAL_ORDER: " << order << endl;
            order = "regular";
        }
    }
    //cout << "BH_FUSER_OPTIMAL_ORDER: " << order << endl;
}

/* Fuse the 'dag' optimally */
void fuse_optimal(bh_ir &bhir, GraphDW &dag)
{
    //The list of edges that we should try to merge
    vector<EdgeW> edges2explore;
    get_edges2explore(dag, edges2explore);
    if(edges2explore.size() == 0)
        return;

    //Check for a preloaded initial condition
    vector<bool> mask(edges2explore.size(), true);
    unsigned int preload_offset=0;
    if(edges2explore.size() > 10)
    {
        const char *t = getenv("BH_FUSER_OPTIMAL_PRELOAD");
        if(t != NULL)
        {
            BOOST_FOREACH(const char &c, string(t))
            {
                mask[preload_offset++] = lexical_cast<bool>(c);
                if(preload_offset == mask.size())
                    break;
            }
            cout << "Preloaded path (" << preload_offset << "): ";
            for(unsigned int j=0; j<preload_offset; ++j)
                cout << mask[j] << ", ";
            cout << endl;
            --preload_offset;
        }
    }

    cout << "FUSER-OPTIMAL: the size of the search space is 2^" << mask.size() << "!" << endl;
    branch_n_bound(bhir, dag, edges2explore, mask, preload_offset);
}

static uint64_t bhir_count=0;
static void manual_merges(GraphDW &dag)
{
    const char *t = getenv("BH_FUSER_OPTIMAL_MERGE");
    if(t == NULL)
        return;
    string s = string(t);

    std::smatch sm;
    std::regex e("\\s*(\\d+):(\\d+)\\+(\\d+),*\\s*");
    while(std::regex_search(s,sm,e))
    {
        assert(sm.size() == 4);
        int dag_id = stoi(sm[1]);
        int v1 = stoi(sm[2]);
        int v2 = stoi(sm[3]);
        if(dag_id == (int)bhir_count)
        {
            cout << "FUSER-OPTIMAL: manual merge of (" << v1 << ", " << v2 << ") in dag " \
                 << dag_id << endl;
            dag.merge_vertices_by_id(v1,v2);
        }
        s = sm.suffix().str();//Iterate to the next match
    }
    dag.remove_cleared_vertices();
}

void do_fusion(bh_ir &bhir)
{
    GraphDW dag;
    from_bhir(bhir, dag);
    fuse_gently(dag);

    manual_merges(dag);

    vector<GraphDW> dags;
    split(dag, dags);
    assert(dag_validate(bhir, dags));
    BOOST_FOREACH(GraphDW &d, dags)
    {
        fuse_gently(d);
        d.transitive_reduction();
        fuse_optimal(bhir, d);
    }
    assert(dag_validate(bhir, dags));
    BOOST_FOREACH(GraphDW &d, dags)
        fill_kernel_list(d.bglD(), bhir.kernel_list);
}

void fuser(bh_ir &bhir, FuseCache &cache)
{
    ++bhir_count;
    if(bhir.kernel_list.size() != 0)
        throw logic_error("The kernel_list is not empty!");

    if(cache.enabled)
    {
        BatchHash hash(bhir.instr_list);
        if(cache.lookup(hash, bhir, bhir.kernel_list))
            return;//Fuse cache hit!
        do_fusion(bhir);
        cache.insert(hash, bhir.kernel_list);
    }
    else
    {
        do_fusion(bhir);
    }
}

