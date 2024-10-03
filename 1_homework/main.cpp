#include "statement.h"
#include <iostream>
#include <sstream>

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

class MainStatement : public Statement {
public:
    MainStatement(unsigned int arguments, unsigned int results, bool pure,
                  const std::vector<std::shared_ptr<Statement>> &poland_operators) : Statement(arguments, results,
                                                                                               pure) {
        operators = poland_operators;
    }

    [[nodiscard]] std::vector<int> apply(std::vector<int> in) const override {
        return in;
    }

private:
    std::vector<std::shared_ptr<Statement>> operators;
};

std::vector<std::string> split(std::string_view str) {
    std::istringstream stream(std::string{str});
    std::string element;
    std::vector<std::string> elements;

    while (stream >> element) {
        elements.push_back(element);
    }

    return elements;
}

std::shared_ptr<Statement> compile(std::string_view str) {
    // TODO: calculate arguments and result
    std::vector<std::string> string_operators{split(str)};
    std::vector<std::shared_ptr<Statement>> operators{};
    std::shared_ptr<Statement> op{nullptr};

    for (std::string_view str_op: string_operators) {
        if (str_op == "+") {
            op = std::make_shared<BinaryOp<std::plus<int>{}>>();
        } else if (str_op == "-") {
            op = std::make_shared<BinaryOp<std::minus<int>{}>>();
        } else if (str_op == "*") {
            op = std::make_shared<BinaryOp<std::multiplies<int>{}>>();
        } else if (str_op == "*") {
            op = std::make_shared<BinaryOp<std::divides<int>{}>>();
        } else {
        }

        operators.push_back(op);
    }

    return std::make_shared<MainStatement>(1, 1, true, operators);
}

int main() {
    auto plus = compile("+");
    auto minus = compile("-");
    auto inc = compile("1 +");

    assert(plus->is_pure() && plus->get_arguments_count() == 2 && plus->get_results_count() == 1);
    assert(inc->is_pure() && inc->get_arguments_count() == 1 && inc->get_results_count() == 1);

    assert(plus->apply({2, 2}) == std::vector{4});
    assert(minus->apply({1, 2, 3}) == std::vector({1, -1}));
}