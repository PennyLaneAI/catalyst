
#pragma once

#include <complex>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "DataView.hpp"
#include "Types.h"

namespace Catalyst::Runtime {
using namespace std;

// string representation of observables
static const unordered_map<ObsId, string> obs_id_to_str = {
    {ObsId::Identity, "Identity"}, {ObsId::PauliX, "PauliX"},     {ObsId::PauliY, "PauliY"},
    {ObsId::PauliZ, "PauliZ"},     {ObsId::Hadamard, "Hadamard"}, {ObsId::Hermitian, "Hermitian"},
};

/**
 * InstructionStrBuilder
 *
 * @brief This class is used by the null device (NullQubit.hpp) whenever the flag to print
 * instructions (print_instructions) is set to true. It is in charge of building string
 * representations of the operations invoked in the aforementioned device interface.
 */

class InstructionStrBuilder {
  private:
    unordered_map<ObsIdType, string>
        obs_id_type_to_str{}; // whenever a new observable is created we store the corresponding
                              // string representation in this hashmap

    /**
     * @brief Template function that returns the string representation of some object.
     */
    template <typename T> string element_to_str(const T &e) { return to_string(e); }

    /**
     * @brief Template specialized to return the string representation of complex numbers.
     */
    template <typename T> string element_to_str(const complex<T> &c)
    {
        ostringstream oss;
        oss << c.real();

        if (c.imag() != 0) {
            // to keep printing output as short as possible, we only print the imaginary part
            // whenever is different from zero;
            if (c.imag() > 0) {
                oss << " + " << c.imag() << "i";
            } else {
                oss << " - " << -1 * c.imag() << "i";
            }
        }
        return oss.str();
    }

    /**
     * @brief Takes as input a vector and returns its string representation. If with_brackets=True,
     * the square brackets will be included to enclose the vector. E.g.
     * - vector_to_string([0,1,2], false) returns "0, 1, 2"
     * - vector_to_string([0,1,2], true) returns "[0, 1, 2]"
     */
    template <typename T>
    string vector_to_string(const vector<T> &v, const bool &with_brackets = true)
    {
        ostringstream oss;
        if (with_brackets)
            oss << "[";

        for (size_t i = 0; i < v.size(); i++) {
            if (i > 0) {
                oss << ", ";
            }
            oss << element_to_str(v[i]);
        }
        if (with_brackets)
            oss << "]";

        return oss.str();
    }

    /**
     * @brief Builds the string representation of DataView
     */
    template <typename T, size_t R> string get_dataview_str(DataView<T, R> &dataview)
    {
        ostringstream oss;
        bool is_first = true; // boolean to help determine where to put commas

        oss << "[";
        for (auto it = dataview.begin(); it != dataview.end(); it++) {
            if (!is_first) {
                oss << ", "; // if is not the first element then we add a comma to separate the
                             // elements
            }
            oss << element_to_str(*it);
            is_first = false;
        }
        oss << "]";
        return oss.str();
    }

  public:
    InstructionStrBuilder() = default;
    ~InstructionStrBuilder() = default;

    /**
     * @brief This method is used to build a string representation of AllocateQubit(),
     * AllocateQubits(), ReleaseQubit(), ReleaseAllQubits(), State(), Measure()
     */
    template <typename T> string get_simple_op_str(const string &name, const T &param)
    {
        ostringstream oss;
        oss << name << "(" << param << ")";
        return oss.str();
    }

    /**
     * @brief This method is used to build a string representation of Expval() and Var()
     */
    string get_op_with_obs_str(const string &name, const ObsIdType &o)
    {
        ostringstream oss;
        oss << name << "(" << get_obs_str(o) << ")";
        return oss.str();
    }

    /**
     * @brief This method is used to build a string representation of Probs(),
     * PartialProbs()
     */
    string get_op_with_view_str(const string &name, DataView<double, 1> &dataview,
                                const std::vector<QubitIdType> &wires = {})
    {
        ostringstream oss;
        oss << name << "(" << get_dataview_str(dataview);
        if (wires.size() > 0) {
            oss << ", wires=" << vector_to_string(wires);
        }
        oss << ")";
        return oss.str();
    }

    /**
     * @brief This method is used to get the string representation of NamedOperation()
     */
    string get_named_op_str(const std::string &name, const std::vector<double> &params,
                            const std::vector<QubitIdType> &wires, bool inverse = false,
                            const std::vector<QubitIdType> &controlled_wires = {},
                            const std::vector<bool> &controlled_values = {},
                            const bool &explicit_wires = true)
    {
        ostringstream oss;
        vector<string> values_to_print; // Store the string representation of the parameters passed
                                        // NamedOperation(). We only consider non-empty vectors to
                                        // preserve printing-output simplicity.

        for (auto p : params)
            values_to_print.push_back(std::to_string(p));

        if (wires.size() > 0) {
            if (explicit_wires) {
                values_to_print.push_back("wires=" + vector_to_string(wires, true));
            }
            else {
                values_to_print.push_back(vector_to_string(wires, false));
            }
        }

        // if inverse is false, we will not print its value
        if (inverse)
            values_to_print.push_back("inverse=true");

        if (controlled_wires.size() > 0)
            values_to_print.push_back("control=" + vector_to_string(controlled_wires));

        if (controlled_values.size() > 0)
            values_to_print.push_back("control_value=" + vector_to_string(controlled_values));

        oss << name << "(";
        for (auto i = 0; i < values_to_print.size(); i++) {
            if (i > 0) {
                oss << ", ";
            }
            oss << values_to_print[i];
        }
        oss << ")";
        return oss.str();
    }

    /**
     * @brief This method is used to get the string representation of MatrixOperation()
     */
    string get_matrix_op_str(const std::vector<std::complex<double>> &matrix,
                             const std::vector<QubitIdType> &wires, bool inverse = false,
                             const std::vector<QubitIdType> &controlled_wires = {},
                             const std::vector<bool> &controlled_values = {},
                             const string &name = "MatrixOperation")
    {
        ostringstream oss;
        vector<string> values_to_print; // Store the string representation of the parameters passed
                                        // NamedOperation(). We only consider non-empty vectors to
                                        // preserve printing-output simplicity.

        values_to_print.emplace_back(vector_to_string(matrix));

        if (wires.size() > 0)
            values_to_print.emplace_back("wires=" + vector_to_string(wires));

        if (inverse)
            values_to_print.push_back("inverse=true");

        if (controlled_wires.size() > 0)
            values_to_print.push_back("control=" + vector_to_string(controlled_wires));

        if (controlled_values.size() > 0)
            values_to_print.push_back("control_value=" + vector_to_string(controlled_values));

        oss << name << "(";
        for (auto i = 0; i < values_to_print.size(); i++) {
            if (i > 0) {
                oss << ", ";
            }
            oss << values_to_print[i];
        }

        oss << ")";
        return oss.str();
    }

    /**
     * @brief Every time Observable() is invoked in the null device interface, we invoke this
     * function to create a new ObsIdType and its corresponding string representation.
     */
    ObsIdType create_obs_str(ObsId obs_id, const std::vector<std::complex<double>> &matrix,
                             const std::vector<QubitIdType> &wires)
    {
        ObsIdType new_id = obs_id_type_to_str.size();

        if (obs_id == ObsId::Hermitian) {
            obs_id_type_to_str.emplace(
                new_id, get_matrix_op_str(matrix, wires, false, {}, {}, "Hermitian"));
        }
        else {
            auto it = obs_id_to_str.find(obs_id);
            if (it != obs_id_to_str.end()) {
                obs_id_type_to_str.emplace(
                    new_id, get_named_op_str(it->second, {}, wires, false, {}, {}, false));
            }
            else {
                RT_FAIL(
                    ("please check obs_id_to_str in file InstructionPrinter. Observation with ID" +
                     to_string(obs_id) + "is not recognized.")
                        .c_str());
            }
        }

        return new_id;
    }

    /**
     * @brief Every time TensorObservable() is invoked in the null device interface, we invoke this
     * function to create a new ObsIdType and its corresponding string representation.
     */
    ObsIdType create_tensor_obs_str(const std::vector<ObsIdType> &obs_keys)
    {
        ostringstream oss;
        for (auto i = 0; i < obs_keys.size(); i++) {
            if (i > 0) {
                oss << " ⊗ ";
            }
            oss << obs_id_type_to_str[obs_keys[i]];
        }
        ObsIdType new_id = obs_id_type_to_str.size();
        obs_id_type_to_str[new_id] = oss.str();
        return new_id;
    }

    /**
     * @brief Every time HamiltonianObservable() is invoked in the null device interface, we invoke
     * this function to create a new ObsIdType and its corresponding string representation.
     */
    ObsIdType create_hamiltonian_obs_str(const std::vector<double> &coeffs,
                                         const std::vector<ObsIdType> &obs_keys)
    {
        RT_FAIL_IF(coeffs.size() != obs_keys.size(),
                   "number of coefficients should match the number of observables");

        ostringstream oss;

        bool is_first = true;
        for (auto i = 0; i < coeffs.size(); i++) {
            if (!is_first) {

                // handle the addition of the terms
                if (coeffs[i] > 0) {
                    oss << " + " << coeffs[i];
                }
                else if (coeffs[i] < 0) {
                    oss << " - " << -1 * coeffs[i]; // a negative sign is manually added so we
                                                    // multiply the coefficient by -1
                }
            }

            if (coeffs[i] != 0) {
                if (is_first)
                    oss << coeffs[i]; // if is the first element then this coefficient is not yet
                                      // added to the string
                oss << "*" << obs_id_type_to_str[obs_keys[i]];
                is_first = false;
            }
        }

        ObsIdType new_id = obs_id_type_to_str.size();
        obs_id_type_to_str[new_id] = oss.str();
        return new_id;
    }

    /**
     * @brief Getter function to retrieve the string representation of the observables we created.
     */
    string get_obs_str(const ObsIdType &o) { return obs_id_type_to_str.at(o); }

    /**
     * @brief This method is used to get the string representation of Sample() and PartialSample()
     */
    string get_samples_str(const string &name, DataView<double, 2> &samples, const size_t &shots,
                           const vector<QubitIdType> &wires = {})
    {
        ostringstream oss;
        bool is_first = true;
        oss << name << "(" << get_dataview_str(samples);

        if (wires.size() > 0) {
            oss << ", wires=" << vector_to_string(wires);
        }

        oss << ", shots=" << shots << ")";

        return oss.str();
    }

    /**
     * @brief This method is used to get the string representation of Counts() and PartialCounts().
     */
    string get_counts_str(const string &name, DataView<double, 1> &vals,
                          DataView<int64_t, 1> &counts, const size_t &shots,
                          const vector<QubitIdType> &wires = {})
    {
        RT_FAIL_IF(vals.size() != counts.size(), "number of eigenvalues does not matches counts");

        ostringstream oss;
        bool is_first = true;
        oss << name << "(";
        oss << "[";
        auto it1 = vals.begin();
        auto it2 = counts.begin();
        for (; it1 != vals.end(); it1++, it2++) {
            if (!is_first) {
                oss << ", ";
            }
            // here we build a substring that represents a pair (eigenval, count)
            oss << "(" << element_to_str(*it1) << ", " << element_to_str(*it2) << ")";
            is_first = false;
        }
        oss << "]"; //

        if (wires.size() > 0) {
            oss << ", wires=" << vector_to_string(wires);
        }

        oss << ", shots=" << shots;

        oss << ")";

        return oss.str();
    }
};
} // namespace Catalyst::Runtime
