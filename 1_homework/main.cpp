#include "statement.h"
#include <iostream>

template<auto func>
class BinaryOp : public Statement {
public:
    BinaryOp() : Statement(2, 1, true) {};

    [[nodiscard]] std::vector<int> apply(std::vector<int> in) const override {
        if (in.size() < 2) {
            std::cout << "BinaryOp Error: not enough arguments" << std::endl;
            exit(1);
        }
        int result = func(in[0], in[1]);
        in.pop_back();
        in.pop_back();
        in.push_back(result);
        return in;
    }
};

template<int number>
class NumberOp : public Statement {
    NumberOp() : Statement(0, 1, true) {};

    [[nodiscard]] std::vector<int> apply(std::vector<int> in) const override {
        in.push_back(number);
        return in;
    }
};

std::vector<std::shared_ptr<Statement>> compile(std::string_view str) {
    std::vector<std::shared_ptr<Statement>> statements{};
    if (str == "+") {
        auto op = std::make_shared<BinaryOp<std::plus<int>{}>>();
        statements.push_back(op);
    }

    return statements;
}

int main() {
    for (auto &el: compile("+")) {
        std::cout << el->is_pure() << " " << el->get_arguments_count() << " " << el->get_results_count() << " "
                  << el->apply({1, 2})[0] << std::endl;
    }
}