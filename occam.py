#!/usr/bin/env python3
import argparse, importlib.util, inspect, sys, time, tracemalloc, math, random, os, gc, dis, ast, textwrap

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    VISUALS_AVAILABLE = True
except ImportError:
    VISUALS_AVAILABLE = False
    print("[!] Warning: 'numpy', 'scipy', or 'matplotlib' not found. Advanced stats and graphs disabled.")

def calculate_mean(data):
    return sum(data) / len(data) if data else 0

def calculate_std(data, mean_val):
    if len(data) < 2: return 0
    variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_cv(mean_val, std_val):
    return (std_val / mean_val) if mean_val > 0 else 0

# --- Static Analysis ---
class CodeInspector:
    @staticmethod
    def calculate_cyclomatic_complexity(source_code):
        try:
            dedented_code = textwrap.dedent(source_code)
            tree = ast.parse(dedented_code)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.ExceptHandler, ast.Assert)):
                    complexity += 1
                elif isinstance(node, (ast.BoolOp)):
                    complexity += len(node.values) - 1
            return complexity
        except: return "?"

    @staticmethod
    def analyze_ast_details(source_code):
        try:
            dedented_code = textwrap.dedent(source_code)
            tree = ast.parse(dedented_code)
        except: return 0, {}
        max_depth = 0
        def _visit_node(node, current_depth):
            nonlocal max_depth
            is_loop = isinstance(node, (ast.For, ast.AsyncFor, ast.While))
            new_depth = current_depth + (1 if is_loop else 0)
            max_depth = max(max_depth, new_depth)
            for child in ast.iter_child_nodes(node):
                _visit_node(child, new_depth)
        _visit_node(tree, 0)
        stats = {"math": 0, "compare": 0, "subscript": 0, "calls": 0}
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.AugAssign)): stats["math"] += 1
            elif isinstance(node, (ast.Compare, ast.BoolOp)): stats["compare"] += 1
            elif isinstance(node, ast.Subscript): stats["subscript"] += 1
            elif isinstance(node, ast.Call): stats["calls"] += 1
        return max_depth, stats

    @staticmethod
    def analyze(func, show_bytecode=False):
        print(f"\n{'-'*65}\nSTATIC ANALYSIS: {func.__name__}\n{'-'*65}")
        try:
            lines, start_line = inspect.getsourcelines(func)
            source_code = "".join(lines)
            loc = len(lines)
            print(f"Lines of Code (LOC):       {loc}")
        except:
            print("Lines of Code (LOC):       N/A (Built-in or Compiled)")
            source_code = ""
        if source_code:
            cc = CodeInspector.calculate_cyclomatic_complexity(source_code)
            risk = "Low"
            if isinstance(cc, int):
                if cc > 10: risk = "Moderate"
                if cc > 20: risk = "High"
                if cc > 50: risk = "Unmaintainable"
            print(f"Cyclomatic Complexity:     {cc} ({risk})")
            nesting, op_stats = CodeInspector.analyze_ast_details(source_code)
            hint = "O(1)?"
            if nesting == 1: hint = "O(N)"
            elif nesting == 2: hint = "O(N^2)"
            elif nesting >= 3: hint = f"O(N^{nesting})"
            print(f"Max Loop Nesting:          {nesting} -> Likely {hint}")
            print(f"Variable Accesses ([]):    {op_stats.get('subscript',0)}")
            print(f"Math Operations:           {op_stats.get('math',0)}")
            print(f"Function Calls:            {op_stats.get('calls',0)}")
        try:
            bytecode = dis.Bytecode(func)
            instructions = list(bytecode)
            count = len(instructions)
            jumps = sum(1 for i in instructions if 'JUMP' in i.opname)
            print(f"Bytecode Instructions:     {count}")
            print(f"Control Flow Jumps:        {jumps}")
            if show_bytecode:
                print(f"\n[Bytecode Dump]")
                dis.dis(func)
        except Exception as e:
            print(f"Bytecode Analysis Failed: {e}")
        print(f"{'-'*65}")

class ComplexityAnalyzer:
    def __init__(self):
        self.wordlist_data = {}
        self.vocab = []

    def load_wordlist(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: Wordlist '{filepath}' not found.")
            sys.exit(1)
        print(f"Loading wordlist from {filepath}...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            words = [line.strip() for line in f if line.strip()]
        unique_chars = set("".join(words))
        self.vocab = list(unique_chars)
        for w in words:
            length = len(w)
            if length not in self.wordlist_data: self.wordlist_data[length] = []
            self.wordlist_data[length].append(w)
        print(f"Loaded {len(words)} inputs. Vocab size: {len(self.vocab)}")

    def generate_input(self, n, input_type, pattern='random'):
        data = None
        if input_type == 'error_nums':
            base = list(range(1, n + 1))
            if n > 1:
                missing_idx = random.randint(0, n - 1)
                dup_source_idx = missing_idx
                while dup_source_idx == missing_idx:
                    dup_source_idx = random.randint(0, n - 1)
                base[missing_idx] = base[dup_source_idx]
            if pattern == 'sorted': base.sort()
            elif pattern == 'reverse': base.sort(reverse=True)
            elif pattern == 'interleaved':
                base.sort()
                half = len(base) // 2
                interleaved = []
                for i in range(half):
                    interleaved.append(base[i])
                    interleaved.append(base[-(i+1)])
                if len(base) % 2 != 0: interleaved.append(base[half])
                base = interleaved
            elif pattern == 'random': random.shuffle(base)
            data = base
        elif input_type == 'int': data = n
        elif input_type == 'list':
            if pattern == 'identical': data = [random.randint(0, 100)] * n
            else:
                data = [random.randint(0, 10000) for _ in range(n)]
                if pattern == 'sorted': data.sort()
                elif pattern == 'reverse': data.sort(reverse=True)
        elif input_type == 'string':
            if pattern == 'identical':
                char = random.choice("abcdefghijklmnopqrstuvwxyz")
                data = char * n
            else:
                chars = random.choices("abcdefghijklmnopqrstuvwxyz", k=n)
                if pattern == 'sorted': chars.sort()
                elif pattern == 'reverse': chars.sort(reverse=True)
                data = "".join(chars)
        elif input_type == 'file':
            if n in self.wordlist_data: data = random.choice(self.wordlist_data[n])
            elif self.vocab: data = "".join(random.choices(self.vocab, k=n))
            else: raise ValueError("Wordlist empty.")
        else: raise ValueError(f"Unknown input type: {input_type}")
        return data
    def analyze_function(self, func_wrapper, input_type, pattern, max_n=5000):
        results = []
        steps = 10
        iterations = 5
        n_values = sorted(list(set([int(max_n * (i / steps)) for i in range(1, steps + 1) if int(max_n * (i / steps)) > 0])))
        print(f"{'N':<8} | {'Time (avg)':<12} | {'Time StdDev':<12} | {'Memory (avg)':<12} | {'Stability (CV)'}")
        print(f"{'-'*75}")
        try:
            warmup_data = self.generate_input(10 if max_n > 100 else 2, input_type, pattern)
            func_wrapper(warmup_data)
        except Exception:
            pass
        for n in n_values:
            times = []
            mems = []
            for _ in range(iterations):
                gc.collect()
                data = self.generate_input(n, input_type, pattern)
                tracemalloc.start()
                start_time = time.perf_counter()
                try:
                    with open(os.devnull, 'w') as fnull:
                        old_stdout = sys.stdout
                        sys.stdout = fnull
                        try: func_wrapper(data)
                        finally: sys.stdout = old_stdout
                except Exception as e:
                    print(f"\n[Error] Crashed on N={n}: {e}")
                    tracemalloc.stop()
                    return results
                end_time = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                times.append(max(end_time - start_time, 1e-10))
                mems.append(max(peak, 1))
            if VISUALS_AVAILABLE:
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_mem = np.mean(mems)
            else:
                avg_time = calculate_mean(times)
                std_time = calculate_std(times, avg_time)
                avg_mem = calculate_mean(mems)
            cv = calculate_cv(avg_time, std_time)
            results.append((n, avg_time, std_time, avg_mem, cv))
            print(f"{n:<8} | {avg_time:.6f}     | {std_time:.6f}     | {int(avg_mem):<12} | {cv:.4f}")
        self._print_slope_summary(results)
        return results

    def _print_slope_summary(self, results):
        if len(results) < 3: return
        n_vals = [math.log(r[0]) for r in results]
        t_vals = [math.log(r[1]) for r in results]
        m_vals = [math.log(r[3]) for r in results]
        time_slope = self._linear_regression(n_vals, t_vals)
        space_slope = self._linear_regression(n_vals, m_vals)
        r_squared = "N/A"
        if VISUALS_AVAILABLE:
            try:
                correlation_matrix = np.corrcoef(n_vals, t_vals)
                correlation_xy = correlation_matrix[0,1]
                r_squared = f"{correlation_xy**2:.4f}"
            except: pass
        print(f"{'-'*75}")
        print(f"Time Slope:  {time_slope:.2f} (R^2: {r_squared}) -> {self._interpret(time_slope)}")
        print(f"Space Slope: {space_slope:.2f} -> {self._interpret(space_slope)}")

    def _linear_regression(self, x, y):
        n = len(x)
        sum_x = sum(x); sum_y = sum(y)
        sum_xy = sum(xi*yi for xi, yi in zip(x, y))
        sum_xx = sum(xi**2 for xi in x)
        denominator = (n * sum_xx) - (sum_x ** 2)
        if denominator == 0: return 0
        return ((n * sum_xy) - (sum_x * sum_y)) / denominator

    def _interpret(self, slope):
        if slope < 0.1: return "O(1) Constant"
        if 0.1 <= slope < 0.85: return "O(log n) Logarithmic"
        if 0.85 <= slope <= 1.15: return "O(n) Linear"
        if 1.15 < slope <= 1.5: return "O(n log n) Quasilinear"
        if 1.5 < slope <= 2.5: return "O(n^2) Quadratic"
        return "O(n^k) Polynomial or Exponential"

    def compare_scripts(self, all_results):
        """Prints a Battle Report comparing multiple scripts."""
        print(f"\n{'='*75}\nBATTLE REPORT: SCRIPT COMPARISON\n{'='*75}")
        print(f"{'Script':<20} | {'Slope':<6} | {'Max Time (s)':<12} | {'Max Mem (B)':<12} | {'Stability (CV)'}")
        print(f"{'-'*75}")
        sorted_scripts = sorted(all_results.items(), key=lambda x: x[1][-1][1])
        for name, data in sorted_scripts:
            if not data: continue
            n_vals = [math.log(r[0]) for r in data]
            t_vals = [math.log(r[1]) for r in data]
            slope = self._linear_regression(n_vals, t_vals)
            max_res = data[-1]
            max_time = max_res[1]
            max_mem = max_res[3]
            avg_cv = sum(r[4] for r in data) / len(data)
            print(f"{name[:20]:<20} | {slope:.2f}   | {max_time:.6f}     | {int(max_mem):<12} | {avg_cv:.4f}")
        print(f"{'-'*75}")
        print("WINNER (Speed): " + sorted_scripts[0][0])

    def plot_performance(self, all_results, filename_prefix="analysis", custom_filename=None):
        if not VISUALS_AVAILABLE:
            print("\n[Info] Install 'matplotlib' and 'scipy' to see graphs.")
            return
        output_file = custom_filename if custom_filename else f"{filename_prefix}_report.png"
        print(f"\nGenerating visualization -> {output_file} ...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Complexity Analysis (with Stability Regions)', fontsize=16)
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
        for idx, (label, data) in enumerate(all_results.items()):
            if not data: continue
            N = np.array([r[0] for r in data])
            T_mean = np.array([r[1] for r in data])
            T_std  = np.array([r[2] for r in data])
            M_mean = np.array([r[3] for r in data])
            color = colors[idx]
            ax1.scatter(N, T_mean, color=color, s=30, alpha=0.8)
            ax1.fill_between(N, T_mean - T_std, T_mean + T_std, color=color, alpha=0.2, label=f"{label} ±1σ")
            self._plot_regression(ax1, N, T_mean, color, label)
            ax2.plot(N, M_mean, 'o--', color=color, label=label, alpha=0.7)
        ax1.set_title('Time Complexity'); ax1.set_xlabel('Input Size (N)'); ax1.set_ylabel('Time (Seconds)'); ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend()
        ax2.set_title('Space Complexity'); ax2.set_xlabel('Input Size (N)'); ax2.set_ylabel('Memory (Bytes)'); ax2.grid(True, linestyle='--', alpha=0.6); ax2.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=100)
        print("Done.")

    def _plot_regression(self, ax, N, T, color, label):
        def power_law(x, a, b): return a * np.power(x, b)
        try:
            popt, _ = curve_fit(power_law, N, T, maxfev=2000)
            a_fit, b_fit = popt
            x_smooth = np.linspace(min(N), max(N), 200)
            y_smooth = power_law(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, '-', color=color, linewidth=2) 
        except: 
            ax.plot(N, T, '-', color=color)

def load_target_function(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
    module_name = os.path.basename(filepath).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and name == "Solution":
            try: instance = obj()
            except: continue
            wrapper = getattr(instance, "run_analysis_entry", None)
            if wrapper and callable(wrapper):
                underlying = [m for m in inspect.getmembers(instance, predicate=inspect.ismethod) if m[0].startswith("_")]
                target = underlying[0][1] if underlying else wrapper
                return wrapper, target
            methods = [m for m in inspect.getmembers(instance, predicate=inspect.ismethod) if not m[0].startswith("_")]
            if methods: return methods[0][1], methods[0][1]
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and obj.__module__ == module_name:
            if name not in ["main", "__main__"] and not name.startswith("_"): return obj, obj
    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Time/Space Complexity (Occam v2)")
    parser.add_argument("files", nargs='+', help="List of Python scripts to analyze")
    parser.add_argument("--wordlist", help="Path to input wordlist file")
    parser.add_argument("--type", choices=['int', 'list', 'string', 'error_nums'], default='int', help="Input data type")
    parser.add_argument("--pattern", choices=['random', 'sorted', 'reverse', 'identical', 'interleaved'], default='random', help="Data pattern")
    parser.add_argument("--max_n", type=int, default=5000, help="Maximum input size (N)")
    parser.add_argument("--output", "-o", help="Custom output filename for the graph")
    parser.add_argument("--dump", action="store_true", help="Dump full bytecode disassembly")
    args = parser.parse_args()
    sys.setrecursionlimit(20000)
    analyzer = ComplexityAnalyzer()
    active_type = args.type
    if args.wordlist:
        analyzer.load_wordlist(args.wordlist)
        active_type = 'file'
    all_results = {}
    for filepath in args.files:
        print(f"\n{'='*75}\nAnalyzing: {filepath}\nPattern:   {args.pattern}\n{'='*75}")
        func_to_run, func_to_inspect = load_target_function(filepath)
        if func_to_run:
            CodeInspector.analyze(func_to_inspect, show_bytecode=args.dump)
            res = analyzer.analyze_function(func_to_run, active_type, args.pattern, args.max_n)
            all_results[os.path.basename(filepath)] = res
        else:
            print(f"Skipping {filepath}: No valid entry point found.")
    if all_results:
        if len(all_results) > 1:
            analyzer.compare_scripts(all_results)
        analyzer.plot_performance(all_results, filename_prefix=f"battle_{args.pattern}", custom_filename=args.output)
