#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <atomic>
#include <memory>
#include <deque>
#include <optional>
#include <thread>
#include <algorithm>
#include <set>
#include <numeric>
#include <cassert>

#include "wsq.hpp"
#include "notifier.hpp"
#include "observer.hpp"
#include "taskflow.hpp"

namespace tf {

/** @class Executor

@brief The executor class to run a taskflow graph.

An executor object manages a set of worker threads and implements 
an efficient work-stealing scheduling algorithm to run a task graph.

*/
class Executor {
  
  struct Worker {
    unsigned id;
    std::mt19937 rdgen { std::random_device{}() };
    WorkStealingQueue<Node*> queue;
    std::optional<Node*> cache;

    int num_executed {0};
  };
    
  struct PerThread {
    Worker* worker {nullptr};
  };

  public:
    
    /**
    @brief constructs the executor with N worker threads
    */
    explicit Executor(unsigned n = std::thread::hardware_concurrency());
    
    /**
    @brief destructs the executor 
    */
    ~Executor();

    /**
    @brief runs the taskflow once
    
    @param taskflow a tf::Taskflow object

    @return a std::future to access the execution state of the taskflow
    */
    std::future<void> run(Taskflow& taskflow);

    /**
    @brief runs the taskflow once and invoke a callback upon completion

    @param taskflow a tf::Taskflow object 
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution state of the taskflow
    */
    template<typename C>
    std::future<void> run(Taskflow& taskflow, C&& callable);

    /**
    @brief runs the taskflow for N times
    
    @param taskflow a tf::Taskflow object
    @param N number of runs

    @return a std::future to access the execution state of the taskflow
    */
    std::future<void> run_n(Taskflow& taskflow, size_t N);

    /**
    @brief runs the taskflow for N times and then invokes a callback

    @param taskflow a tf::Taskflow 
    @param N number of runs
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution state of the taskflow
    */
    template<typename C>
    std::future<void> run_n(Taskflow& taskflow, size_t N, C&& callable);

    /**
    @brief runs the taskflow multiple times until the predicate becomes true and 
           then invokes a callback

    @param taskflow a tf::Taskflow 
    @param pred a boolean predicate to return true for stop

    @return a std::future to access the execution state of the taskflow
    */
    template<typename P>
    std::future<void> run_until(Taskflow& taskflow, P&& pred);

    /**
    @brief runs the taskflow multiple times until the predicate becomes true and 
           then invokes the callback

    @param taskflow a tf::Taskflow 
    @param pred a boolean predicate to return true for stop
    @param callable a callable object to be invoked after this run

    @return a std::future to access the execution state of the taskflow
    */
    template<typename P, typename C>
    std::future<void> run_until(Taskflow& taskflow, P&& pred, C&& callable);

    /**
    @brief wait for all pending graphs to complete
    */
    void wait_for_all();

    /**
    @brief queries the number of worker threads (can be zero)

    @return the number of worker threads
    */
    size_t num_workers() const;
    
    /**
    @brief constructs an observer to inspect the activities of worker threads

    Each executor manages at most one observer at a time through std::unique_ptr.
    Createing multiple observers will only keep the lastest one.
    
    @tparam Observer observer type derived from tf::ExecutorObserverInterface
    @tparam ArgsT... argument parameter pack

    @param args arguments to forward to the constructor of the observer
    
    @return a raw pointer to the observer associated with this executor
    */
    template<typename Observer, typename... Args>
    Observer* make_observer(Args&&... args);
    
    /**
    @brief removes the associated observer
    */
    void remove_observer();

    /**
    @brief queries the id of the caller thread in this executor
    */
    std::optional<unsigned> this_worker_id() const;

  private:
   
    std::condition_variable _topology_cv;
    std::mutex _topology_mutex;
    std::mutex _queue_mutex;

    unsigned _num_topologies {0};
    
    // scheduler field
    std::vector<Worker> _workers;
    std::vector<Notifier::Waiter> _waiters;
    std::vector<std::thread> _threads;

    WorkStealingQueue<Node*> _queue;

    std::atomic<size_t> _num_actives {0};
    std::atomic<size_t> _num_thieves {0};
    std::atomic<bool>   _done        {0};

    Notifier _notifier;
    
    std::unique_ptr<ExecutorObserverInterface> _observer;
    
    unsigned _find_victim(unsigned);

    PerThread& _per_thread() const;

    bool _wait_for_task(Worker&, std::optional<Node*>&);
    
    void _spawn(unsigned);
    void _exploit_task(Worker&, std::optional<Node*>&);
    void _explore_task(Worker&, std::optional<Node*>&);
    void _schedule(Node*, bool);
    void _schedule(PassiveVector<Node*>&);
    void _invoke(Worker&, Node*);
    void _invoke_static_work(Worker&, Node*);
    void _invoke_dynamic_work(Worker&, Node*, Subflow&);
    void _set_up_module_node(Node*);
    void _set_up_topology(Topology*);
    void _tear_down_topology(Topology**); 
    void _increment_topology();
    void _decrement_topology();
    void _decrement_topology_and_notify();
};

// Constructor
inline Executor::Executor(unsigned N) : 
  _workers  {N},
  _waiters  {N},
  _notifier {_waiters} {
  
  if(N == 0) {
    TF_THROW(Error::EXECUTOR, "no workers to execute the graph");
  }

  _spawn(N);
}

// Destructor
inline Executor::~Executor() {
  
  // wait for all topologies to complete
  wait_for_all();
  
  // shut down the scheduler
  _done = true;
  _notifier.notify(true);
  
  for(auto& t : _threads){
    t.join();
  } 
}

// Function: num_workers
inline size_t Executor::num_workers() const {
  return _workers.size();
}

// Function: _per_thread
inline Executor::PerThread& Executor::_per_thread() const {
  thread_local PerThread pt;
  return pt;
}

// Function: this_worker_id
inline std::optional<unsigned> Executor::this_worker_id() const {
  if(auto worker = _per_thread().worker; worker) {
    return worker->id;
  }
  else {
    return std::nullopt;
  }
}

// Procedure: _spawn
inline void Executor::_spawn(unsigned N) {

  // Lock to synchronize all workers before creating _worker_maps
  for(unsigned i=0; i<N; ++i) {

    _workers[i].id = i;

    _threads.emplace_back([this] (Worker& w) -> void {

      PerThread& pt = _per_thread();  
      pt.worker = &w;
    
      std::optional<Node*> t;
      
      // must use 1 as condition instead of !done
      while(1) {
        
        // execute the tasks.
        _exploit_task(w, t);

        // wait for tasks
        if(_wait_for_task(w, t) == false) {
          break;
        }
      }
      
    }, std::ref(_workers[i]));     
  }
}

// Function: _find_victim
inline unsigned Executor::_find_victim(unsigned thief) {
  
  /*unsigned l = 0;
  unsigned r = _workers.size() - 1;
  unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(
    _workers[thief].rdgen
  );

  // try to look for a task from other workers
  for(unsigned i=0; i<_workers.size(); ++i){

    if((thief == vtm && !_queue.empty()) ||
       (thief != vtm && !_workers[vtm].queue.empty())) {
      return vtm;
    }

    if(++vtm; vtm == _workers.size()) {
      vtm = 0;
    }
  } */

  // try to look for a task from other workers
  for(unsigned vtm=0; vtm<_workers.size(); ++vtm){
    if((thief == vtm && !_queue.empty()) ||
       (thief != vtm && !_workers[vtm].queue.empty())) {
      return vtm;
    }
  }

  return static_cast<unsigned>(_workers.size());
}

// Function: _explore_task
inline void Executor::_explore_task(Worker& thief, std::optional<Node*>& t) {
  
  //assert(_workers[thief].queue.empty());
  assert(!t);

  const unsigned l = 0;
  const unsigned r = static_cast<unsigned>(_workers.size()) - 1;

  const size_t F = (_workers.size() + 1) << 1;
  const size_t Y = 100;

  size_t f = 0;
  size_t y = 0;

  // explore
  while(!_done) {
  
    unsigned vtm = std::uniform_int_distribution<unsigned>{l, r}(thief.rdgen);
      
    t = (vtm == thief.id) ? _queue.steal() : _workers[vtm].queue.steal();

    if(t) {
      break;
    }

    if(f++ > F) {
      if(std::this_thread::yield(); y++ > Y) {
        break;
      }
    }

    /*if(auto vtm = _find_victim(thief); vtm != _workers.size()) {
      t = (vtm == thief) ? _queue.steal() : _workers[vtm].queue.steal();
      // successful thief
      if(t) {
        break;
      }
    }
    else {
      if(f++ > F) {
        if(std::this_thread::yield(); y++ > Y) {
          break;
        }
      }
    }*/
  }

}

// Procedure: _exploit_task
inline void Executor::_exploit_task(Worker& worker, std::optional<Node*>& t) {
  
  assert(!worker.cache);

  if(t) {
    if(_num_actives.fetch_add(1) == 0 && _num_thieves == 0) {
      _notifier.notify(false);
    }

    Topology *tpg = (*t)->_topology;
    Node* prev_parent {(*t)->_parent};
    worker.num_executed = 1;

    do {
      // Only joined subflow will enter block
      // Flush the num_executed if encountering a different subflow
      //if((*t)->_parent != prev_parent) {
      //  if(prev_parent == nullptr) {
      //    (*t)->_topology->_join_counter.fetch_sub(worker.num_executed);
      //  }
      //  else {
      //    auto ret = prev_parent->_join_counter.fetch_sub(worker.num_executed);
      //    if(ret == worker.num_executed) {
      //      _schedule(prev_parent, false);
      //    }
      //  }
      //  worker.num_executed = 1;
      //  prev_parent = (*t)->_parent;
      //}

      _invoke(worker, *t);

      if(worker.cache) {
        t = *worker.cache;
        worker.cache = std::nullopt;
      }
      else {
        t = worker.queue.pop();
        if(t) {
          // We only increment the counter when poping task from queue (NOT including cache!)
          if((*t)->_parent == prev_parent) {
            worker.num_executed ++;
          }
          // joined subflow
          else {
            if(prev_parent == nullptr) {
              // still have tasks so the topology join counter can't be zero
              (*t)->_topology->_join_counter.fetch_sub(worker.num_executed);
            }
            else {
              auto ret = prev_parent->_join_counter.fetch_sub(worker.num_executed);
              if(ret == worker.num_executed) {
                //_schedule(prev_parent, false);
                worker.queue.push(prev_parent);
              }
            }
            worker.num_executed = 1;
            prev_parent = (*t)->_parent;
          }
        }
        else {
          // If no more local tasks!
          if(prev_parent == nullptr) {
            if(tpg->_join_counter.fetch_sub(worker.num_executed) == worker.num_executed) {
              // TODO: Store tpg in local variable not in worker
              _tear_down_topology(&tpg);
              if(tpg != nullptr) {
                t = worker.queue.pop();
                if(t) {
                  worker.num_executed = 1;
                }
              }
            }
          }
          else {
            if(prev_parent->_join_counter.fetch_sub(worker.num_executed) == worker.num_executed) {
              t = prev_parent;
              worker.num_executed = 1;
              prev_parent = prev_parent->_parent;
            }
          }
        }
      }
    } while(t);

    --_num_actives;
  }
}

// Function: _wait_for_task
inline bool Executor::_wait_for_task(Worker& worker, std::optional<Node*>& t) {

  wait_for_task:

  assert(!t);

  ++_num_thieves;

  explore_task:

  if(_explore_task(worker, t); t) {
    if(auto N = _num_thieves.fetch_sub(1); N == 1) {
      _notifier.notify(false);
    }
    return true;
  }

  auto waiter = &_waiters[worker.id];

  _notifier.prepare_wait(waiter);
  
  //if(auto vtm = _find_victim(me); vtm != _workers.size()) {
  if(!_queue.empty()) {

    _notifier.cancel_wait(waiter);
    //t = (vtm == me) ? _queue.steal() : _workers[vtm].queue.steal();

    if(t = _queue.steal(); t) {
      if(auto N = _num_thieves.fetch_sub(1); N == 1) {
        _notifier.notify(false);
      }
      return true;
    }
    else {
      goto explore_task;
    }
  }

  if(_done) {
    _notifier.cancel_wait(waiter);
    _notifier.notify(true);
    --_num_thieves;
    return false;
  }

  if(_num_thieves.fetch_sub(1) == 1 && _num_actives) {
    _notifier.cancel_wait(waiter);
    goto wait_for_task;
  }
    
  // Now I really need to relinguish my self to others
  _notifier.commit_wait(waiter);

  return true;
}

// Function: make_observer    
template<typename Observer, typename... Args>
Observer* Executor::make_observer(Args&&... args) {
  // use a local variable to mimic the constructor 
  auto tmp = std::make_unique<Observer>(std::forward<Args>(args)...);
  tmp->set_up(_workers.size());
  _observer = std::move(tmp);
  return static_cast<Observer*>(_observer.get());
}

// Procedure: remove_observer
inline void Executor::remove_observer() {
  _observer.reset();
}

// Procedure: _schedule
// The main procedure to schedule a give task node.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(Node* node, bool bypass) {
  
  //assert(_workers.size() != 0);
  
  // module node need another initialization
  if(node->_module && !node->_module->empty() && !node->_has_state(Node::SPAWNED)) {
    _set_up_module_node(node);
  }
  
  // caller is a worker to this pool
  if(auto worker = _per_thread().worker; worker != nullptr) {
    if(!bypass) {
      worker->queue.push(node);
    }
    else {
      assert(!worker->cache);
      worker->cache = node;
    }
    return;
  }

  // other threads
  {
    std::scoped_lock lock(_queue_mutex);
    _queue.push(node);
  }

  _notifier.notify(false);
}

// Procedure: _schedule
// The main procedure to schedule a set of task nodes.
// Each task node has two types of tasks - regular and subflow.
inline void Executor::_schedule(PassiveVector<Node*>& nodes) {

  //assert(_workers.size() != 0);
  
  // We need to cacth the node count to avoid accessing the nodes
  // vector while the parent topology is removed!
  const auto num_nodes = nodes.size();
  
  if(num_nodes == 0) {
    return;
  }

  for(auto node : nodes) {
    if(node->_module && !node->_module->empty() && !node->_has_state(Node::SPAWNED)) {
      _set_up_module_node(node);
    }
  }

  // worker thread
  if(auto worker = _per_thread().worker; worker != nullptr) {
    for(size_t i=0; i<num_nodes; ++i) {
      worker->queue.push(nodes[i]);
    }
    return;
  }
  
  // other threads
  {
    std::scoped_lock lock(_queue_mutex);
    for(size_t k=0; k<num_nodes; ++k) {
      _queue.push(nodes[k]);
    }
  }

  if(num_nodes >= _workers.size()) {
    _notifier.notify(true);
  }
  else {
    for(size_t k=0; k<num_nodes; ++k) {
      _notifier.notify(false);
    }
  }
}


// Procedure: _invoke
inline void Executor::_invoke(Worker& worker, Node* node) {

  //assert(_workers.size() != 0);

  // Here we need to fetch the num_successors first to avoid the invalid memory
  // access caused by topology clear.
  const auto num_successors = node->num_successors();
  
  // condition task
  if(node->_work.index() == Node::CONDITION_WORK) {

    if(node->_has_state(Node::BRANCH)) {
      node->_join_counter = node->num_strong_dependents();
    }
    else {
      node->_join_counter = node->num_dependents();
    }

    if(size_t id = std::get<Node::ConditionWork>(node->_work)(); id < num_successors) {
      node->_successors[id]->_join_counter.store(0);
      _schedule(node->_successors[id], true);
    }
    return ;
  }
  // static task
  // The default node work type. We only need to execute the callback if any.
  else if(auto index=node->_work.index(); index == Node::STATIC_WORK) {
    if(node->_module != nullptr) {
      bool first_time = !node->_has_state(Node::SPAWNED);
      _invoke_static_work(worker, node);
      if(first_time) {
        return ;
      }
    }
    else {
      _invoke_static_work(worker, node);
    }
  } 
  // dynamic task
  else if (index == Node::DYNAMIC_WORK){

    // Clear the subgraph before the task execution
    if(!node->_has_state(Node::SPAWNED)) {
      if(node->_subgraph) {
        node->_subgraph->clear();
      }
      else {
        node->_subgraph.emplace();
      }
    }
   
    Subflow fb(*(node->_subgraph));

    _invoke_dynamic_work(worker, node, fb);
    
    // Need to create a subflow if first time & subgraph is not empty 
    if(!node->_has_state(Node::SPAWNED)) {
      node->_set_state(Node::SPAWNED);
      if(!node->_subgraph->empty()) {
        // For storing the source nodes
        PassiveVector<Node*> src; 

        for(auto n: node->_subgraph->_nodes) {

          n->_topology = node->_topology;
          n->_set_up_join_counter();
          
          //n->_set_state(Node::SUBTASK);

          if(!fb.detached()) {
            n->_parent = node;
          }

          if(n->num_dependents() == 0) {
            src.push_back(n);
          }
        }

        const bool join = fb.joined();
        if(!join) {
          // Detach mode
          node->_topology->_join_counter.fetch_add(src.size());         
        }
        else {
          // Join mode
          node->_join_counter.fetch_add(src.size());

          // spawned node needs another second-round execution
          if(node->_parent == nullptr) {
            node->_topology->_join_counter.fetch_add(1);
          }
          else {
            node->_parent->_join_counter.fetch_add(1);
          }
        }

        _schedule(src);

        if(join) {
          return;
        }
      } // End of first time 
    }
  } // End of DynamicWork -----------------------------------------------------
  

  // We MUST recover the dependency since subflow is a condition node can go back (cyclic)
  // This must be done before scheduling the successors, otherwise this might cause 
  // race condition on the _dependents
  if(node->_has_state(Node::BRANCH)) {
    // If this is a case node, we need to deduct condition predecessors
    node->_join_counter = node->num_strong_dependents();
  }
  else {
    node->_join_counter = node->num_dependents();
  }

  node->_unset_state(Node::SPAWNED);

  // At this point, the node storage might be destructed.
  Node* cache {nullptr};
  size_t num_spawns {0};

  auto& c = (node->_parent) ? node->_parent->_join_counter : node->_topology->_join_counter;

  for(size_t i=0; i<num_successors; ++i) {
    if(--(node->_successors[i]->_join_counter) == 0) {
      if(cache) {
        if(num_spawns == 0) {
          c.fetch_add(num_successors);
        }
        num_spawns++;
        _schedule(cache, false);
      }
      cache = node->_successors[i];
    }
  }

  if(num_spawns) {
    worker.num_executed += (node->_successors.size() - num_spawns);
  }

  if(cache) {
    _schedule(cache, true);
  }
}

// Procedure: _invoke_static_work
inline void Executor::_invoke_static_work(Worker& worker, Node* node) {
  if(_observer) {
    _observer->on_entry(worker.id, TaskView(node));
    std::invoke(std::get<Node::StaticWork>(node->_work));
    _observer->on_exit(worker.id, TaskView(node));
  }
  else {
    std::invoke(std::get<Node::StaticWork>(node->_work));
  }
}

// Procedure: _invoke_dynamic_work
inline void Executor::_invoke_dynamic_work(Worker& worker, Node* node, Subflow& sf) {
  if(_observer) {
    _observer->on_entry(worker.id, TaskView(node));
    std::invoke(std::get<Node::DynamicWork>(node->_work), sf);
    _observer->on_exit(worker.id, TaskView(node));
  }
  else {
    std::invoke(std::get<Node::DynamicWork>(node->_work), sf);
  }
}

// Function: run
inline std::future<void> Executor::run(Taskflow& f) {
  return run_n(f, 1, [](){});
}

// Function: run
template <typename C>
std::future<void> Executor::run(Taskflow& f, C&& c) {
  static_assert(std::is_invocable<C>::value);
  return run_n(f, 1, std::forward<C>(c));
}

// Function: run_n
inline std::future<void> Executor::run_n(Taskflow& f, size_t repeat) {
  return run_n(f, repeat, [](){});
}

// Function: run_n
template <typename C>
std::future<void> Executor::run_n(Taskflow& f, size_t repeat, C&& c) {
  return run_until(f, [repeat]() mutable { return repeat-- == 0; }, std::forward<C>(c));
}

// Function: run_until    
template<typename P>
std::future<void> Executor::run_until(Taskflow& f, P&& pred) {
  return run_until(f, std::forward<P>(pred), [](){});
}

// Function: _set_up_topology
inline void Executor::_set_up_topology(Topology* tpg) {
  
  tpg->_sources.clear();
  
  // scan each node in the graph and build up the links
  for(auto node : tpg->_taskflow._graph._nodes) {

    node->_topology = tpg;
    node->_clear_state();

    if(node->num_dependents() == 0) {
      tpg->_sources.push_back(node);
    }

    int join_counter = 0;
    for(auto p : node->_dependents) {
      if(p->_work.index() == Node::CONDITION_WORK) {
        node->_set_state(Node::BRANCH);
      }
      else {
        join_counter++;
      }
    }

    node->_join_counter.store(join_counter, std::memory_order_relaxed);
  }

  tpg->_join_counter.store(tpg->_sources.size(), std::memory_order_relaxed);
}

// Function: _tear_down_topology
inline void Executor::_tear_down_topology(Topology** tpg) {

  auto &f = (*tpg)->_taskflow;

  //assert(&tpg == &(f._topologies.front()));

  // case 1: we still need to run the topology again
  if(!std::invoke((*tpg)->_pred)) {
    //tpg->_recover_num_sinks();

    assert((*tpg)->_join_counter == 0);
    (*tpg)->_join_counter = (*tpg)->_sources.size();

    _schedule((*tpg)->_sources); 
  }
  // case 2: the final run of this topology
  else {
    
    if((*tpg)->_call != nullptr) {
      std::invoke((*tpg)->_call);
    }

    f._mtx.lock();

    // If there is another run (interleave between lock)
    if(f._topologies.size() > 1) {

      assert((*tpg)->_join_counter == 0);

      // Set the promise
      (*tpg)->_promise.set_value();
      f._topologies.pop_front();
      f._mtx.unlock();
      
      // decrement the topology but since this is not the last we don't notify
      _decrement_topology();

      *tpg = &(f._topologies.front());

      _set_up_topology(*tpg);
      _schedule((*tpg)->_sources);

      //f._topologies.front()._bind(f._graph);
      //*tpg = &(f._topologies.front());

      //assert(f._topologies.front()._join_counter == 0);

      //f._topologies.front()._join_counter = f._topologies.front()._sources.size();

      //_schedule(f._topologies.front()._sources);
    }
    else {
      assert(f._topologies.size() == 1);

      // Need to back up the promise first here becuz taskflow might be 
      // destroy before taskflow leaves
      auto p {std::move((*tpg)->_promise)};

      f._topologies.pop_front();

      f._mtx.unlock();

      // We set the promise in the end in case taskflow leaves before taskflow
      p.set_value();

      _decrement_topology_and_notify();

      // Reset topology so caller can stop execution
      *tpg = nullptr;
    }
  }
}

// Function: run_until
template <typename P, typename C>
std::future<void> Executor::run_until(Taskflow& f, P&& pred, C&& c) {

  // Predicate must return a boolean value
  static_assert(std::is_invocable_v<C> && std::is_invocable_v<P>);
  
  _increment_topology();

  // Special case of predicate
  if(f.empty() || std::invoke(pred)) {
    std::promise<void> promise;
    promise.set_value();
    _decrement_topology_and_notify();
    return promise.get_future();
  }
  

  
  //// Special case of zero workers requires:
  ////  - iterative execution to avoid stack overflow
  ////  - avoid execution of last_work
  //if(_workers.size() == 0) {
  //  
  //  Topology tpg(f, std::forward<P>(pred), std::forward<C>(c));

  //  // Clear last execution data & Build precedence between nodes and target
  //  tpg._bind(f._graph);

  //  std::stack<Node*> stack;

  //  do {
  //    _schedule_unsync(tpg._sources, stack);
  //    while(!stack.empty()) {
  //      auto node = stack.top();
  //      stack.pop();
  //      _invoke_unsync(node, stack);
  //    }
  //    tpg._recover_num_sinks();
  //  } while(!std::invoke(tpg._pred));

  //  if(tpg._call != nullptr) {
  //    std::invoke(tpg._call);
  //  }

  //  tpg._promise.set_value();
  //  
  //  _decrement_topology_and_notify();
  //  
  //  return tpg._promise.get_future();
  //}

  // Multi-threaded execution.
  bool run_now {false};
  Topology* tpg;
  std::future<void> future;
  
  {
    std::scoped_lock lock(f._mtx);

    // create a topology for this run
    tpg = &(f._topologies.emplace_back(f, std::forward<P>(pred), std::forward<C>(c)));
    future = tpg->_promise.get_future();
   
    if(f._topologies.size() == 1) {
      run_now = true;
      //tpg->_bind(f._graph);
      //_schedule(tpg->_sources);
    }
  }
  
  // Notice here calling schedule may cause the topology to be removed sonner 
  // before the function leaves.
  if(run_now) {
    _set_up_topology(tpg);
    _schedule(tpg->_sources);
  }

  return future;
}

// Procedure: _increment_topology
inline void Executor::_increment_topology() {
  std::scoped_lock<std::mutex> lock(_topology_mutex);
  ++_num_topologies;
}

// Procedure: _decrement_topology_and_notify
inline void Executor::_decrement_topology_and_notify() {
  std::scoped_lock<std::mutex> lock(_topology_mutex);
  if(--_num_topologies == 0) {
    _topology_cv.notify_all();
  }
}

// Procedure: _decrement_topology
inline void Executor::_decrement_topology() {
  std::scoped_lock lock(_topology_mutex);
  --_num_topologies;
}

// Procedure: wait_for_all
inline void Executor::wait_for_all() {
  std::unique_lock lock(_topology_mutex);
  _topology_cv.wait(lock, [&](){ return _num_topologies == 0; });
}


// Procedure: _set_up_module_node
inline void Executor::_set_up_module_node(Node* node) {

  node->_work = [node=node, this] () {

    // second time to enter this context
    if(node->_has_state(Node::SPAWNED)) {
      return ;
    }

    // first time to enter this context
    node->_set_state(Node::SPAWNED);

    PassiveVector<Node*> src;

    for(auto n: node->_module->_graph._nodes) {

      n->_topology = node->_topology;
      n->_parent = node;
      n->_set_up_join_counter();

      if(n->num_dependents() == 0) {
        src.push_back(n);
      }
    }

    node->_join_counter.fetch_add(src.size());
    
    //auto worker = _per_thread().worker;
    //
    //if(worker != nullptr) {
    if(node->_parent == nullptr) {
      node->_topology->_join_counter.fetch_add(1);
    }
    else {
      node->_parent->_join_counter.fetch_add(1);
    }
    //}
    //// TODO (twhuang 01/02/20): will this happen?
    //else {
    //  node->_topology->_join_counter.fetch_add(src.size());         
    //}
    
    // TODO: error if src is empty?
    _schedule(src);
  };
}

}  // end of namespace tf -----------------------------------------------------


