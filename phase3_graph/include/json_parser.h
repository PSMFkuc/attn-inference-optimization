#ifndef PHASE3_JSON_PARSER_H
#define PHASE3_JSON_PARSER_H

#include "graph.h"
#include <string>

// ===========================================================================
// JSON parser: read a model config file and build a ComputeGraph
// ===========================================================================
// We use a minimal hand-rolled JSON parser to avoid dependencies.
// Supported format:
//   {
//     "nodes": [
//       { "id": "...", "op": "...", "inputs": [...], "params": {...} },
//       ...
//     ]
//   }
// Params are optional key-value pairs (float values only).
// ===========================================================================

class JsonParser {
public:
    // Parse a JSON string and populate the graph
    static bool parse(const std::string& json, ComputeGraph& graph);

    // Parse from file
    static bool parse_file(const std::string& path, ComputeGraph& graph);
};

#endif // PHASE3_JSON_PARSER_H
