/*******************************************************************************
 * This file is part of KaHyPar.
 * ... (license text remains unchanged) ...
 ******************************************************************************/

 #pragma once

 #include <limits>
 #include <vector>  // For std::vector
 
 #include <kahypar/definitions.h>
 #include <kahypar/partition/context.h>
 #include <kahypar-resources/utils/randomize.h>
 #include "kahypar-resources/datastructure/fast_reset_flag_array.h"
 
 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Weffc++"
 

 #include <WHFC/datastructure/flow_hypergraph_builder.h>
 #include <WHFC/datastructure/node_border.h>
 #include <WHFC/datastructure/flow_hypergraph.h>
 #include <WHFC/datastructure/queue.h>
 #include <WHFC/datastructure/distance_reachable_sets.h>
 
 #pragma GCC diagnostic pop
 
 namespace kahypar {
 namespace whfcInterface {
   class FlowHypergraphExtractor {
     private:
      static constexpr bool enable_heavy_assert = false;
    
     public:
      static constexpr HypernodeID invalid_node = std::numeric_limits<HypernodeID>::max();
      static constexpr PartitionID invalid_part = std::numeric_limits<PartitionID>::max();
      static constexpr bool debug = false;
    
      FlowHypergraphExtractor(const Hypergraph& hg, const Context& context)
          : flow_hg_builder(hg.initialNumNodes(), hg.initialNumEdges(), hg.initialNumPins()),
            nodeIDMap(hg.initialNumNodes() + 2, whfc::invalidNode),
            visitedNode(hg.initialNumNodes() + 2),
            visitedHyperedge(hg.initialNumEdges()),
            queue(hg.initialNumNodes() + 2) {
        removeHyperedgesWithPinsOutsideRegion = context.partition.objective == Objective::cut;
      }
 
   struct AdditionalData {
     whfc::Node source;
     whfc::Node target;
     whfc::Flow baseCut;
     whfc::Flow cutAtStake;
   };
 
   // Changed whfc::DistanceFromCut to std::vector<whfc::HopDistance>
   AdditionalData run(const Hypergraph& hg, const Context& context, std::vector<HyperedgeID>& cut_hes,
                      const PartitionID _b0, const PartitionID _b1, std::vector<whfc::HopDistance>& distanceFromCut) {
     whfc::HopDistance hop_distance_delta = context.local_search.hyperflowcutter.use_distances_from_cut ? 1 : 0;
 
     AdditionalData result = { whfc::invalidNode, whfc::invalidNode, 0, 0 };
     reset(hg, _b0, _b1);
 
     auto[maxW0, maxW1] = flowHyperGraphPartSizes(context, hg);
     HypernodeWeight w0 = 0, w1 = 0;
     std::shuffle(cut_hes.begin(), cut_hes.end(), Randomize::instance().getGenerator());
 
     result.source = whfc::Node::fromOtherValueType(queue.queueEnd());
     queue.push(globalSourceID);
     queue.reinitialize();
     flow_hg_builder.addNode(whfc::NodeWeight(0));
     BreadthFirstSearch(hg, b0, b1, cut_hes, w0, maxW0, result.source, -hop_distance_delta, distanceFromCut);
 
     result.target = whfc::Node::fromOtherValueType(queue.queueEnd());
     queue.push(globalTargetID);
     queue.reinitialize();
     flow_hg_builder.addNode(whfc::NodeWeight(0));
     BreadthFirstSearch(hg, b1, b0, cut_hes, w1, maxW1, result.target, hop_distance_delta, distanceFromCut);
 
     for (const HyperedgeID e : cut_hes) {
       ASSERT(!visitedHyperedge[e], "Cut HE list contains duplicates");
       if (canHyperedgeBeDropped(hg, e))
         continue;
       result.cutAtStake += hg.edgeWeight(e);
       bool connectToSource = false;
       bool connectToTarget = false;
       visitedHyperedge.set(e);
       flow_hg_builder.startHyperedge(hg.edgeWeight(e));
       for (const HypernodeID v : hg.pins(e)) {
         if (visitedNode[v]) {
           flow_hg_builder.addPin(nodeIDMap[v]);
         } else {
           connectToSource |= hg.inPart(v, b0);
           connectToTarget |= hg.inPart(v, b1);
           if (connectToSource && connectToTarget)
             break;
         }
       }
       if (connectToSource && connectToTarget) {
         flow_hg_builder.removeCurrentHyperedge();
         result.baseCut += hg.edgeWeight(e);
       } else {
         ASSERT(flow_hg_builder.currentHyperedgeSize() != 0, "he in cut but has no pin in flow hg, except maybe one terminal");
         if (connectToSource)
           flow_hg_builder.addPin(result.source);
         if (connectToTarget)
           flow_hg_builder.addPin(result.target);
       }
     }
 
     HEAVY_REFINEMENT_ASSERT([&]() {
       for (HypernodeID u : hg.nodes()) {
         if (hg.isFixedVertex(u) && global2local(u) != whfc::invalidNode) {
           DBG << "Fixed vertex" << V(u) << V(global2local(u)) << "included in the snapshot for FlowCutter refinement.";
           return false;
         }
       }
       return true;
     }());
 
     whfc::NodeWeight ws(hg.partWeight(b0) - w0), wt(hg.partWeight(b1) - w1);
     if (ws == 0 || wt == 0) {
       result.cutAtStake = 0;
       result.baseCut = 0;
     } else {
       flow_hg_builder.nodeWeight(result.source) = ws;
       flow_hg_builder.nodeWeight(result.target) = wt;
       flow_hg_builder.finalize();
     }
     return result;
   }
 
   auto localNodeIDs() const { return boost::irange<whfc::Node>(whfc::Node(0), whfc::Node::fromOtherValueType(queue.queueEnd())); }
   whfc::Node global2local(const HypernodeID x) const { ASSERT(x != globalSourceID && x != globalTargetID); return nodeIDMap[x]; }
   HypernodeID local2global(const whfc::Node x) const { return queue.elementAt(x); }
 
  private:
   bool canHyperedgeBeDropped(const Hypergraph& hg, const HyperedgeID e) {
     return removeHyperedgesWithPinsOutsideRegion && hg.hasPinsInOtherBlocks(e, b0, b1);
   }
 
   inline void visitNode(const HypernodeID v, const Hypergraph& hg, HypernodeWeight& w) {
     nodeIDMap[v] = whfc::Node::fromOtherValueType(queue.queueEnd());
     ASSERT(nodeIDMap[v] == flow_hg_builder.numNodes());
     flow_hg_builder.addNode(whfc::NodeWeight(hg.nodeWeight(v)));
     queue.push(v);
     visitedNode.set(v);
     w += hg.nodeWeight(v);
   }
 
   // Changed whfc::DistanceFromCut to std::vector<whfc::HopDistance>
   void BreadthFirstSearch(const Hypergraph& hg, const PartitionID myBlock, const PartitionID otherBlock,
                           const std::vector<HyperedgeID>& cut_hes, HypernodeWeight& w,
                           double sizeConstraint, const whfc::Node myTerminal,
                           whfc::HopDistance d_delta, std::vector<whfc::HopDistance>& distanceFromCut) {
     whfc::HopDistance d = d_delta;
     for (const HyperedgeID e : cut_hes) {
       for (const HypernodeID u: hg.pins(e)) {
         if (!visitedNode[u] && hg.inPart(u, myBlock)
             && hg.nodeWeight(u) + w <= sizeConstraint && !hg.isFixedVertex(u)) {
           visitNode(u, hg, w);
           distanceFromCut[nodeIDMap[u]] = d;
         }
       }
     }
 
     while (!queue.empty()) {
       if (queue.currentLayerEmpty()) {
         queue.finishNextLayer();
         d += d_delta;
       }
       HypernodeID u = queue.pop();
       for (const HyperedgeID e : hg.incidentEdges(u)) {
         if (!hg.hasPinsInPart(e, otherBlock) && hg.pinCountInPart(e, myBlock) > 1 &&
             (!canHyperedgeBeDropped(hg, e)) && !visitedHyperedge[e]) {
           visitedHyperedge.set(e);
           flow_hg_builder.startHyperedge(hg.edgeWeight(e));
           bool connectToTerminal = false;
           for (const HypernodeID v : hg.pins(e)) {
             if (hg.inPart(v, myBlock)) {
               if (!visitedNode[v] && w + hg.nodeWeight(v) <= sizeConstraint && !hg.isFixedVertex(v)) {
                 visitNode(v, hg, w);
                 distanceFromCut[nodeIDMap[v]] = d;
               }
 
               if (visitedNode[v])
                 flow_hg_builder.addPin(nodeIDMap[v]);
               else
                 connectToTerminal = true;
             }
           }
           if (connectToTerminal)
             flow_hg_builder.addPin(myTerminal);
         }
       }
     }
 
     d += d_delta;
     distanceFromCut[myTerminal] = d;
   }
 
  public:
   whfc::FlowHypergraphBuilder flow_hg_builder;
 
  private:
   PartitionID b0 = invalid_part, b1 = invalid_part;
   HypernodeID globalSourceID = invalid_node, globalTargetID = invalid_node;
   std::vector<whfc::Node> nodeIDMap;
   ds::FastResetFlagArray<> visitedNode;
   ds::FastResetFlagArray<> visitedHyperedge;
   using Queue = LayeredQueue<HypernodeID>;
   Queue queue;
   bool removeHyperedgesWithPinsOutsideRegion = false;
 
   void reset(const Hypergraph& hg, const PartitionID _b0, const PartitionID _b1) {
     b0 = _b0;
     b1 = _b1;
     flow_hg_builder.clear();
     visitedNode.reset();
     visitedHyperedge.reset();
     queue.clear();
 
     globalSourceID = hg.initialNumNodes();
     globalTargetID = hg.initialNumNodes() + 1;
   }
 
   std::pair<double, double> flowHyperGraphPartSizes(const Context& context, const Hypergraph& hg) const {
     double mw0 = 0.0, mw1 = 0.0;
     double a = context.local_search.hyperflowcutter.snapshot_scaling;
 
     if (context.local_search.hyperflowcutter.flowhypergraph_size_constraint == FlowHypergraphSizeConstraint::part_weight_fraction) {
       mw0 = a * hg.partWeight(b0);
       mw1 = a * hg.partWeight(b1);
     } else if (context.local_search.hyperflowcutter.flowhypergraph_size_constraint == FlowHypergraphSizeConstraint::max_part_weight_fraction) {
       mw0 = a * context.partition.max_part_weights[b0];
       mw1 = a * context.partition.max_part_weights[b1];
     } else if (context.local_search.hyperflowcutter.flowhypergraph_size_constraint == FlowHypergraphSizeConstraint::scaled_max_part_weight_fraction_minus_opposite_side) {
       double scale;
       if (!context.partition.use_individual_part_weights) {
         scale = 1.0 + a * std::min(0.05, context.partition.epsilon);
       } else {
         scale = 1.0 + (a-1) * std::min(0.05, context.partition.adjusted_epsilon_for_individual_part_weights);
       }
       mw0 = scale * context.partition.perfect_balance_part_weights[b1] - hg.partWeight(b1);
       mw1 = scale * context.partition.perfect_balance_part_weights[b0] - hg.partWeight(b0);
     } else {
       throw std::runtime_error("Unknown flow hypergraph size constraint option");
     }
 
     mw0 = std::min(mw0, hg.partWeight(b0) * 0.9999);
     mw0 = std::max(mw0, 0.0);
     mw1 = std::min(mw1, hg.partWeight(b1) * 0.9999);
     mw1 = std::max(mw1, 0.0);
     return std::make_pair(mw0, mw1);
   }
 };
 }
 }