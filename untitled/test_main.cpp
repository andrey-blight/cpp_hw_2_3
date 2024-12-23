#include <iostream>
#include <cassert>
#include <random>
#include <regex>
#include <fstream>

using namespace std;
static constexpr size_t N = 36, M = 84;

template<int N, bool FAST>
struct SelectIntType;

template<int N>
struct SelectIntType<N, false> {
    static_assert(N == 8 || N == 16 || N == 32 || N == 64,
                  "N must be one of 8, 16, 32, or 64 when FAST == false");

    using type = std::conditional_t<
            N == 8, int8_t,
            std::conditional_t<
                    N == 16, int16_t,
                    std::conditional_t<
                            N == 32, int32_t,
                            int64_t
                    >
            >
    >;
};

// === Реализация для FAST == true ===
template<int N>
struct SelectIntType<N, true> {
    static_assert(N > 0, "N must be a positive integer when FAST == true");

    using type = std::conditional_t<
            (N <= 8), int_fast8_t,
            std::conditional_t<
                    (N <= 16), int_fast16_t,
                    std::conditional_t<
                            (N <= 32), int_fast32_t,
                            int_fast64_t
                    >
            >
    >;
};

// === Шаблонный класс Fixed<N, K> ===
template<int N, int K, bool FAST = false>
struct Fixed {
    static_assert(N > 0, "N must be a positive integer");

    using int_t = typename SelectIntType<N, FAST>::type;

    constexpr Fixed(int v) : v(v << K) {}

    constexpr Fixed(float f) : v(f * (1 << K)) {}

    constexpr Fixed(double f) : v(f * (1 << K)) {}

    constexpr Fixed() : v(0) {}

    template<int N2, int K2, bool FAST2>
    constexpr Fixed(const Fixed<N2, K2, FAST2> &other) {
        if (K2 > K) {
            int_t converted_value = static_cast<int_t>(other.v) >> (K2 - K);
            v = converted_value;
        } else if (K2 < K) {
            int_t converted_value = static_cast<int_t>(other.v) << (K - K2);
            v = converted_value;
        } else {
            v = other.v;
        }
    }

    static constexpr Fixed from_raw(int_t x) {
        Fixed ret;
        ret.v = x;
        return ret;
    }

    int_t v;

    // === Операторы сравнения ===
    auto operator<=>(const Fixed &) const = default;

    std::partial_ordering operator<=>(double other) const {
        double fixed_as_double = v / static_cast<double>(1 << K);
        if (fixed_as_double < other) return std::partial_ordering::less;
        if (fixed_as_double > other) return std::partial_ordering::greater;
        return std::partial_ordering::equivalent;
    }

    // Приведение Fixed к double
    explicit operator double() const {
        return v / static_cast<double>(1 << K);
    }

    bool operator==(const Fixed &) const = default;

    static std::string to_str() {
        if (FAST) {
            return "FAST_FIXED(" + std::to_string(N) + "," + std::to_string(K) + ")";
        }
        return "FIXED(" + std::to_string(N) + "," + std::to_string(K) + ")";
    }
};

template<int N1, int K1, bool FAST1, int N2, int K2, bool FAST2>
Fixed<N1, K1, FAST1> operator+(Fixed<N1, K1, FAST1> a, Fixed<N2, K2, FAST2> b) {
    using int_t1 = typename SelectIntType<N1, FAST1>::type;
    using int_t2 = typename SelectIntType<N2, FAST2>::type;
    int_t1 a_val = a.v;
    int_t2 b_val = b.v;
    return Fixed<N1, K1, FAST1>::from_raw(a_val + b_val);
}

template<int N1, int K1, bool FAST1, int N2, int K2, bool FAST2>
Fixed<N1, K1, FAST1> operator-(Fixed<N1, K1, FAST1> a, Fixed<N2, K2, FAST2> b) {
    using int_t1 = typename SelectIntType<N1, FAST1>::type;
    using int_t2 = typename SelectIntType<N2, FAST2>::type;
    int_t1 a_val = a.v;
    int_t2 b_val = b.v;
    return Fixed<N1, K1, FAST1>::from_raw(a_val - b_val);
}

template<int N1, int K1, bool FAST1, int N2, int K2, bool FAST2>
Fixed<N1, K1, FAST1> operator*(Fixed<N1, K1, FAST1> a, Fixed<N2, K2, FAST2> b) {
    using int_t1 = typename SelectIntType<N1, FAST1>::type;
    using int_t2 = typename SelectIntType<N2, FAST2>::type;
    int_t1 a_val = a.v;
    int_t2 b_val = b.v;

    return Fixed<N1, K1, FAST1>::from_raw(((int64_t) a_val * b_val) >> K1);
}

template<int N1, int K1, bool FAST1, int N2, int K2, bool FAST2>
Fixed<N1, K1, FAST1> operator/(Fixed<N1, K1, FAST1> a, Fixed<N2, K2, FAST2> b) {
    using int_t1 = typename SelectIntType<N1, FAST1>::type;
    using int_t2 = typename SelectIntType<N2, FAST2>::type;
    int_t1 a_val = a.v;
    int_t2 b_val = b.v;

    return Fixed<N1, K1, FAST1>::from_raw(((int64_t) a_val << K1) / b_val);
}

template<int N1, int K1, bool FAST1, int N2, int K2, bool FAST2>
Fixed<N1, K1, FAST1> &operator-=(Fixed<N1, K1, FAST1> &a, Fixed<N2, K2, FAST2> b) {
    return a = a - b;
}

template<int N1, int K1, bool FAST1, int N2, int K2, bool FAST2>
Fixed<N1, K1, FAST1> &operator+=(Fixed<N1, K1, FAST1> &a, Fixed<N2, K2, FAST2> b) {
    return a = a + b;
}

template<int N1, int K1, bool FAST1, int N2, int K2, bool FAST2>
Fixed<N1, K1, FAST1> &operator*=(Fixed<N1, K1, FAST1> &a, Fixed<N2, K2, FAST2> b) {
    return a = a * b;
}

template<int N, int K>
ostream &operator<<(ostream &out, Fixed<N, K> x) {
    return out << static_cast<double>(x);
}

template<typename P, typename V, typename V_flow>
class Simulation {
private:
//    static constexpr size_t N = 36, M = 84;
// constexpr size_t N = 14, M = 5;
    static constexpr size_t T = 1'000'000;
    static constexpr std::array<pair<int, int>, 4> deltas{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

    static constexpr P inf = Fixed<64, 16>::from_raw(std::numeric_limits<int32_t>::max());
    static constexpr P eps = Fixed<64, 16>::from_raw(deltas.size());

    int dirs[N][M]{};

    P rho[256];
    P g = 0.1;

    P p[N][M]{}, old_p[N][M];

    template<typename V_t>
    struct VectorField {
        array<V_t, deltas.size()> v[N][M];

        V_t &add(int x, int y, int dx, int dy, V_t dv) {
            return get(x, y, dx, dy) += dv;
        }

        V_t &get(int x, int y, int dx, int dy) {
            size_t i = ranges::find(deltas, pair(dx, dy)) - deltas.begin();
            assert(i < deltas.size());
            return v[x][y][i];
        }
    };

    VectorField<V> velocity{};
    VectorField<V_flow> velocity_flow{};
    int last_use[N][M]{};
    int UT = 0;


    mt19937 rnd{1337};

// char field[N][M + 1] = {
//     "#####",
//     "#.  #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#.# #",
//     "#...#",
//     "#####",
//     "#   #",
//     "#   #",
//     "#   #",
//     "#####",
// };

    char field[N][M + 1] = {
            "####################################################################################",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                       .........                                  #",
            "#..............#            #           .........                                  #",
            "#..............#            #           .........                                  #",
            "#..............#            #           .........                                  #",
            "#..............#            #                                                      #",
            "#..............#            #                                                      #",
            "#..............#            #                                                      #",
            "#..............#            #                                                      #",
            "#..............#............#                                                      #",
            "#..............#............#                                                      #",
            "#..............#............#                                                      #",
            "#..............#............#                                                      #",
            "#..............#............#                                                      #",
            "#..............#............#                                                      #",
            "#..............#............#                                                      #",
            "#..............#............#                                                      #",
            "#..............#............################                     #                 #",
            "#...........................#....................................#                 #",
            "#...........................#....................................#                 #",
            "#...........................#....................................#                 #",
            "##################################################################                 #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "#                                                                                  #",
            "####################################################################################",
    };

public:
    using P_t = P;
    using V_t = V;
    using V_flow_t = V_flow;

    tuple<V_flow, bool, pair<int, int>> propagate_flow(int x, int y, V_flow lim);

    void propagate_stop(int x, int y, bool force = false);

    Fixed<32, 16> random01();

    V move_prob(int x, int y);

    void swap_with(int x, int y);

    bool propagate_move(int x, int y, bool is_first);

    void start(P rho_space, P rho_dot, P g_my, char my_filed[N][M + 1]);

};

template<typename P, typename V, typename V_flow>
void Simulation<P, V, V_flow>::swap_with(int x, int y) {
    char type;
    P cur_p;
    array<V, deltas.size()> v{};

    swap(field[x][y], type);
    swap(p[x][y], cur_p);
    swap(velocity.v[x][y], v);
}

template<typename P, typename V, typename V_flow>
tuple<V_flow, bool, pair<int, int>> Simulation<P, V, V_flow>::propagate_flow(int x, int y, V_flow lim) {
    last_use[x][y] = UT - 1;
    V_flow ret = 0;
    for (auto [dx, dy]: deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] != '#' && last_use[nx][ny] < UT) {
            V cap = velocity.get(x, y, dx, dy);
            V_flow flow = velocity_flow.get(x, y, dx, dy);
            if (flow == static_cast<V_flow>(cap)) {
                continue;
            }
            // assert(v >= velocity_flow.get(x, y, dx, dy));
            auto vp = min(static_cast<V_flow>(lim), static_cast<V_flow>(static_cast<V_flow>(cap) - flow));
            if (last_use[nx][ny] == UT - 1) {
                velocity_flow.add(x, y, dx, dy, vp);
                last_use[x][y] = UT;
                // cerr << x << " " << y << " -> " << nx << " " << ny << " " << vp << " / " << lim << "\n";
                return {vp, 1, {nx, ny}};
            }
            auto [t, prop, end] = propagate_flow(nx, ny, vp);
            ret += t;
            if (prop) {
                velocity_flow.add(x, y, dx, dy, t);
                last_use[x][y] = UT;
                // cerr << x << " " << y << " -> " << nx << " " << ny << " " << t << " / " << lim << "\n";
                return {t, end != pair(x, y), end};
            }
        }
    }
    last_use[x][y] = UT;
    return {ret, false, {0, 0}};
}

template<typename P, typename V, typename V_flow>
Fixed<32, 16> Simulation<P, V, V_flow>::random01() {
    return Fixed<32, 16>::from_raw((rnd() & ((1 << 16) - 1)));
}


template<typename P, typename V, typename V_flow>
void Simulation<P, V, V_flow>::propagate_stop(int x, int y, bool force) {
    if (!force) {
        bool stop = true;
        for (auto [dx, dy]: deltas) {
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) > 0) {
                stop = false;
                break;
            }
        }
        if (!stop) {
            return;
        }
    }
    last_use[x][y] = UT;
    for (auto [dx, dy]: deltas) {
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT || velocity.get(x, y, dx, dy) > 0) {
            continue;
        }
        propagate_stop(nx, ny);
    }
}

template<typename P, typename V, typename V_flow>
V Simulation<P, V, V_flow>::move_prob(int x, int y) {
    V sum = 0;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
            continue;
        }
        auto v = velocity.get(x, y, dx, dy);
        if (v < 0) {
            continue;
        }
        sum += v;
    }
    return sum;
}


template<typename P, typename V, typename V_flow>
bool Simulation<P, V, V_flow>::propagate_move(int x, int y, bool is_first) {
    last_use[x][y] = UT - is_first;
    bool ret = false;
    int nx = -1, ny = -1;
    do {
        std::array<V, deltas.size()> tres;
        V sum = 0;
        for (size_t i = 0; i < deltas.size(); ++i) {
            auto [dx, dy] = deltas[i];
            int nx = x + dx, ny = y + dy;
            if (field[nx][ny] == '#' || last_use[nx][ny] == UT) {
                tres[i] = sum;
                continue;
            }
            auto v = velocity.get(x, y, dx, dy);
            if (v < 0) {
                tres[i] = sum;
                continue;
            }
            sum += v;
            tres[i] = sum;
        }

        if (sum == 0) {
            break;
        }

        V p = static_cast<V>(random01()) * sum;
        size_t d = std::ranges::upper_bound(tres, p) - tres.begin();

        auto [dx, dy] = deltas[d];
        nx = x + dx;
        ny = y + dy;
        assert(velocity.get(x, y, dx, dy) > 0 && field[nx][ny] != '#' && last_use[nx][ny] < UT);

        ret = (last_use[nx][ny] == UT - 1 || propagate_move(nx, ny, false));
    } while (!ret);
    last_use[x][y] = UT;
    for (size_t i = 0; i < deltas.size(); ++i) {
        auto [dx, dy] = deltas[i];
        int nx = x + dx, ny = y + dy;
        if (field[nx][ny] != '#' && last_use[nx][ny] < UT - 1 && velocity.get(x, y, dx, dy) < 0) {
            propagate_stop(nx, ny);
        }
    }
    if (ret) {
        if (!is_first) {
            swap_with(x, y);
            swap_with(nx, ny);
            swap_with(x, y);
        }
    }
    return ret;
}

template<typename P, typename V, typename V_flow>
void Simulation<P, V, V_flow>::start(P rho_space, P rho_dot, P g_my, char my_filed[N][M + 1]) {
    rho[' '] = rho_space;
    rho['.'] = rho_dot;
    g = g_my;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            field[i][j] = my_filed[i][j];
        }
    }

    for (size_t x = 0; x < N; ++x) {
        for (size_t y = 0; y < M; ++y) {
            if (field[x][y] == '#')
                continue;
            for (auto [dx, dy]: deltas) {
                dirs[x][y] += (field[x + dx][y + dy] != '#');
            }
        }
    }

    for (size_t i = 0; i < T; ++i) {

        P total_delta_p = 0;
        // Apply external forces
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                if (field[x + 1][y] != '#')
                    velocity.add(x, y, 1, 0, static_cast<V>(g));
            }
        }

        // Apply forces from p
        memcpy(old_p, p, sizeof(p));
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                for (auto [dx, dy]: deltas) {
                    int nx = x + dx, ny = y + dy;
                    if (field[nx][ny] != '#' && old_p[nx][ny] < old_p[x][y]) {
                        auto delta_p = old_p[x][y] - old_p[nx][ny];
                        auto force = delta_p;
                        auto &contr = velocity.get(nx, ny, -dx, -dy);
                        if (static_cast<P>(contr) * static_cast<P>(rho[(int) field[nx][ny]]) >= force) {
                            contr -= static_cast<V>(force / rho[(int) field[nx][ny]]);
                            continue;
                        }
                        force -= static_cast<P>(contr) * static_cast<P>(rho[(int) field[nx][ny]]);
                        contr = 0;
                        velocity.add(x, y, dx, dy, static_cast<V>(force / rho[(int) field[x][y]]));
                        p[x][y] -= force / static_cast<P>(dirs[x][y]);
                        total_delta_p -= force / static_cast<P>(dirs[x][y]);
                    }
                }
            }
        }

        // Make flow from velocities
        velocity_flow = {};
        bool prop = false;
        do {
            UT += 2;
            prop = 0;
            for (size_t x = 0; x < N; ++x) {
                for (size_t y = 0; y < M; ++y) {
                    if (field[x][y] != '#' && last_use[x][y] != UT) {
                        auto [t, local_prop, _] = propagate_flow(x, y, 1);
                        if (t > static_cast<V_flow>(0)) {
                            prop = 1;
                        }
                    }
                }
            }
        } while (prop);

        // Recalculate p with kinetic energy
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] == '#')
                    continue;
                for (auto [dx, dy]: deltas) {
                    auto old_v = velocity.get(x, y, dx, dy);
                    auto new_v = velocity_flow.get(x, y, dx, dy);
                    if (old_v > 0) {
                        assert(new_v <= static_cast<V_flow>(old_v));
                        velocity.get(x, y, dx, dy) = static_cast<V>(new_v);
                        auto force = static_cast<P>(static_cast<V_flow>(old_v) - new_v) * rho[(int) field[x][y]];
                        if (field[x][y] == '.')
                            force *= static_cast<P>(0.8);
                        if (field[x + dx][y + dy] == '#') {
                            p[x][y] += force / static_cast<P>(dirs[x][y]);
                            total_delta_p += force / static_cast<P>(dirs[x][y]);
                        } else {
                            p[x + dx][y + dy] += force / static_cast<P>(dirs[x + dx][y + dy]);
                            total_delta_p += force / static_cast<P>(dirs[x + dx][y + dy]);
                        }
                    }
                }
            }
        }

        UT += 2;
        prop = false;
        for (size_t x = 0; x < N; ++x) {
            for (size_t y = 0; y < M; ++y) {
                if (field[x][y] != '#' && last_use[x][y] != UT) {
                    if (static_cast<V>(random01()) < move_prob(x, y)) {
                        prop = true;
                        propagate_move(x, y, true);
                    } else {
                        propagate_stop(x, y, true);
                    }
                }
            }
        }

        if (prop) {
            cout << "Tick " << i << ":\n";
            for (size_t x = 0; x < N; ++x) {
                cout << field[x] << "\n";
            }
        }
    }
}

#define TYPES DOUBLE, FIXED(32,16)
#define FLOAT float
#define DOUBLE double
#define FIXED(N, K) Fixed<N,K>
#define FAST_FIXED(N, K) Fixed<N,K,true>

using Types = std::tuple<TYPES >;

template<typename TypeList, template<typename, typename, typename> class Template>
struct Combinator;

template<typename... Types, template<typename, typename, typename> class Template>
struct Combinator<tuple<Types...>, Template> {
private:

    template<typename T, typename... Rest>
    struct Generate {
        using type = decltype(std::tuple_cat(
                std::declval<std::tuple<Template<T, Types, Types>...>>(),
                typename Generate<Rest...>::type{}
        ));
    };

    template<typename T>
    struct Generate<T> {
        using type = std::tuple<Template<T, Types, Types>...>;
    };

public:
    using type = typename Generate<Types...>::type;
};

using AllClasses = Combinator<Types, Simulation>::type;

template<typename TupleType, std::size_t... I>
auto create_classes(std::index_sequence<I...>) {
    return std::make_tuple(typename std::tuple_element<I, TupleType>::type()...);
}

auto build() {
    constexpr std::size_t N = std::tuple_size<AllClasses>::value;
    return create_classes<AllClasses>(std::make_index_sequence<N>{});
}

template<typename P, typename V, typename V_flow>
size_t generate_type_hash() {
    return (typeid(P).hash_code() + 123) ^ (typeid(V).hash_code() + 234) ^ (typeid(V_flow).hash_code() + 193);
}

template<std::size_t I = 0, typename TupleType>
void fill_class_map(TupleType &tuple, std::unordered_map<size_t, void *> &map) {
    if constexpr (I < std::tuple_size<TupleType>::value) {
        using ClassType = typename std::tuple_element<I, TupleType>::type;
        size_t type_hash = generate_type_hash<typename ClassType::P_t, typename ClassType::V_t, typename ClassType::V_flow_t>();
        map[type_hash] = &std::get<I>(tuple);
        fill_class_map<I + 1>(tuple, map);
    }
}

template<typename T>
struct TypeName;

template<>
struct TypeName<DOUBLE> {
    static std::string to_str() { return "DOUBLE"; }
};

template<>
struct TypeName<FLOAT> {
    static std::string to_str() { return "FLOAT"; }
};

template<int N, int K, bool FAST>
struct TypeName<Fixed<N, K, FAST>> {
    static std::string to_str() { return Fixed<N, K, FAST>::to_str(); }
};

template<typename... Types>
std::unordered_map<std::string, size_t> create_type_map() {
    std::unordered_map<std::string, size_t> type_map;

    auto insert_type = [&type_map](auto &&type_instance) {
        using T = std::decay_t<decltype(type_instance)>;
        std::string type_name = TypeName<T>::to_str();
        type_map[type_name] = typeid(T).hash_code();
    };

    (insert_type(Types{}), ...);

    return type_map;
}

template<std::size_t I = 0, typename TupleType>
bool
call_start_if_match(TupleType &tuple, size_t target_hash, double rho_space, double rho_dot, double g,
                    char field[N][M + 1]) {
    if constexpr (I < std::tuple_size<TupleType>::value) {
        using ClassType = typename std::tuple_element<I, TupleType>::type;
        size_t type_hash = generate_type_hash<typename ClassType::P_t, typename ClassType::V_t, typename ClassType::V_flow_t>();

        if (type_hash == target_hash) {
            std::cout << "Найден класс: Simulation<"
                      << typeid(typename ClassType::P_t).name() << ", "
                      << typeid(typename ClassType::V_t).name() << ", "
                      << typeid(typename ClassType::V_flow_t).name() << ">\n";

            std::get<I>(tuple).start(rho_space, rho_dot, g, field);
            return true;
        }

        return call_start_if_match<I + 1>(tuple, target_hash, rho_space, rho_dot, g, field);
    }
    return false;
}

int main(int argc, char *argv[]) {
    // Создаем мапу "str_type" -> type
    auto type_map = create_type_map<TYPES >();

    double rho_space = 0.01, rho_dot = 1000, g = 0.1;

    size_t p, v, v_flow;
    char field[N][M + 1];
    // Находим типы P, V, V_flow из параметров
    for (int i = 0; i < argc; ++i) {
        string el{argv[i]};
        if (el.starts_with("--p-type=")) {
            el = el.substr(9, el.size() - 9);
            if (type_map.find(el) == type_map.end()) {
                throw runtime_error("Bad type");
            }
            p = type_map[el];
        } else if (el.starts_with("--v-type=")) {
            el = el.substr(9, el.size() - 9);
            if (type_map.find(el) == type_map.end()) {
                throw runtime_error("Bad type");
            }
            v = type_map[el];
        } else if (el.starts_with("--v-flow-type=")) {
            el = el.substr(14, el.size() - 14);
            if (type_map.find(el) == type_map.end()) {
                throw runtime_error("Bad type");
            }
            v_flow = type_map[el];
        } else if (el.starts_with("--file=")) {
            el = el.substr(7, el.size() - 7);

            std::ifstream file(el);
            if (!file) {
                std::cerr << "Не удалось открыть файл!" << std::endl;
                return 1;
            }

            file >> rho_space >> rho_dot >> g;

            size_t row = 0;
            std::string line;
            std::getline(file,line); // сбрасываем лишний перевод строки
            while (std::getline(file, line) && row < N) {
                std::strncpy(field[row], line.c_str(), M);
                field[row][M] = '\0';
                ++row;
            }
        }
    }

    // Создаем все возможные классы Simulation<P,V,V_flow>
    auto all_classes = build();

    // Создаем мапу с хешем шаблона и самим типом шаблона
    std::unordered_map<size_t, void *> class_map;
    fill_class_map(all_classes, class_map);

    size_t target_hash = (p + 123) ^ (v + 234) ^ (v_flow + 193);
    // Смотрим есть ли такой класс с таким шаблоном и запускаем
    if (!call_start_if_match(all_classes, target_hash, rho_space, rho_dot, g, field)) {
        throw runtime_error("Bad type");
    }
}
