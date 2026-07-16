#include "json_parser.h"
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstdio>
#include <stdexcept>

// ===========================================================================
// Minimal hand-rolled JSON parser
// ===========================================================================
// Why not use nlohmann/json or simdjson?
//   Phase 3 is about understanding graph execution, not JSON parsing.
//   A hand-rolled parser keeps dependencies minimal and is ~100 lines.
//
// What this parser supports:
//   - Objects: { "key": value, ... }
//   - Arrays:  [value, ...]
//   - Strings: "text"
//   - Numbers: 123, 1.5, -0.5
//   - Booleans and null are NOT supported (not needed for graph config)
// ===========================================================================

class Tokenizer {
public:
    explicit Tokenizer(const std::string& s) : s_(s), pos_(0) {}

    char peek() {
        skip_whitespace();
        return pos_ < s_.size() ? s_[pos_] : '\0';
    }

    char next() {
        skip_whitespace();
        return pos_ < s_.size() ? s_[pos_++] : '\0';
    }

    void expect(char c) {
        char got = next();
        if (got != c) {
            throw std::runtime_error(
                std::string("Expected '") + c + "' but got '" + got + "' at pos " +
                std::to_string(pos_));
        }
    }

    std::string read_string() {
        expect('"');
        std::string result;
        while (pos_ < s_.size() && s_[pos_] != '"') {
            result += s_[pos_++];
        }
        expect('"');
        return result;
    }

    float read_number() {
        skip_whitespace();
        size_t start = pos_;
        if (pos_ < s_.size() && s_[pos_] == '-') pos_++;
        while (pos_ < s_.size() && (std::isdigit(s_[pos_]) || s_[pos_] == '.')) pos_++;
        return std::stof(s_.substr(start, pos_ - start));
    }

    void skip_whitespace() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
               s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == ','))
            pos_++;
    }

private:
    std::string s_;
    size_t pos_;
};

// ===========================================================================
// Parse
// ===========================================================================

static void parse_node(Tokenizer& tok, ComputeGraph& graph) {
    Node node;
    tok.expect('{');

    while (tok.peek() != '}') {
        std::string key = tok.read_string();
        tok.expect(':');

        if (key == "id") {
            node.id = tok.read_string();
        } else if (key == "op") {
            node.op_type = tok.read_string();
        } else if (key == "inputs") {
            tok.expect('[');
            while (tok.peek() != ']') {
                node.inputs.push_back(tok.read_string());
            }
            tok.expect(']');
        } else if (key == "params") {
            tok.expect('{');
            while (tok.peek() != '}') {
                std::string pkey = tok.read_string();
                tok.expect(':');
                float pval = tok.read_number();
                node.params[pkey] = pval;
            }
            tok.expect('}');
        }
    }
    tok.expect('}');

    // Auto-generate output name if not specified
    if (node.outputs.empty()) {
        node.outputs.push_back(node.id);
    }

    graph.add_node(node);
}

bool JsonParser::parse(const std::string& json, ComputeGraph& graph) {
    try {
        Tokenizer tok(json);
        tok.expect('{');

        while (tok.peek() != '}') {
            std::string key = tok.read_string();
            tok.expect(':');

            if (key == "nodes") {
                tok.expect('[');
                while (tok.peek() != ']') {
                    parse_node(tok, graph);
                }
                tok.expect(']');
            } else {
                // Skip unknown keys
                if (tok.peek() == '"') tok.read_string();
                else if (tok.peek() == '{') {
                    int depth = 1;
                    while (depth > 0 && tok.peek() != '\0') {
                        char c = tok.next();
                        if (c == '{') depth++;
                        if (c == '}') depth--;
                    }
                }
            }
        }
        tok.expect('}');
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[JSON Parser Error] %s\n", e.what());
        return false;
    }
}

bool JsonParser::parse_file(const std::string& path, ComputeGraph& graph) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[JSON Parser] Cannot open file: %s\n", path.c_str());
        return false;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    return parse(buf.str(), graph);
}
