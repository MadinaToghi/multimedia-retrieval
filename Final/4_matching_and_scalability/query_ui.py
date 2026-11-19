"""Tkinter interface for choosing a query mesh and browsing ranked matches."""

import base64
import io
import queue
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

FINAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINAL_DIR / "1_read_data"))

import MMR_utility as mmru


MAX_CACHED_PREVIEWS = 12


class QueryWindow:
    def __init__(
        self,
        query_database,
        descriptor_weights,
        database_path,
        source_path,
        features_path,
        top_k=5,
        seed=None,
        search_index=None,
    ):
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.query_database = query_database
        self.descriptor_weights = descriptor_weights
        self.database_path = Path(database_path)
        self.source_path = Path(source_path)
        self.features_path = Path(features_path)
        self.seed = seed
        self.search_index = search_index
        self.result_queue = queue.Queue()
        self.result_cards = []
        self.preview_cache = OrderedDict()
        self._preview_refresh_after = None
        self._closed = False

        self.root = tk.Tk()
        self.root.title("MMR Shape Query")
        self.root.geometry("1380x900")
        self.root.minsize(1000, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        if not self.source_path.is_dir():
            messagebox.showerror(
                "Source database not found",
                f"Cannot find the source mesh database:\n{self.source_path}",
                parent=self.root,
            )
            self.root.destroy()
            self.root = None
            return

        self.meshes_by_category = {}
        for mesh_path in mmru.find_obj_files(self.source_path):
            self.meshes_by_category.setdefault(mesh_path.parent.name, []).append(
                mesh_path
            )
        self.categories = sorted(self.meshes_by_category)

        self._build_widgets(top_k)

    def _build_widgets(self, top_k):
        tk, ttk = self.tk, self.ttk
        selector = ttk.LabelFrame(
            self.root, text="Choose a query mesh", padding=8
        )
        selector.pack(fill="x", padx=10, pady=(10, 5))

        self.source_tabs = ttk.Notebook(selector)
        self.source_tabs.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        selector.columnconfigure(0, weight=1)
        selector.rowconfigure(0, weight=1)

        database_tab = ttk.Frame(self.source_tabs, padding=8)
        self.source_tabs.add(database_tab, text="From ShapeDatabase_INFOMR")
        self.category_var = tk.StringVar()
        self.mesh_var = tk.StringVar()
        ttk.Label(database_tab, text="Category").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self.category_combo = ttk.Combobox(
            database_tab,
            textvariable=self.category_var,
            values=self.categories,
            state="readonly",
            width=28,
        )
        self.category_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        self.category_combo.bind(
            "<<ComboboxSelected>>", self._category_changed
        )
        ttk.Label(database_tab, text="Mesh").grid(
            row=0, column=2, sticky="w", padx=(0, 6)
        )
        self.mesh_combo = ttk.Combobox(
            database_tab,
            textvariable=self.mesh_var,
            state="readonly",
            width=28,
        )
        self.mesh_combo.grid(row=0, column=3, sticky="ew")
        database_tab.columnconfigure(1, weight=1)
        database_tab.columnconfigure(3, weight=1)
        if self.categories:
            self.category_combo.current(0)
            self._category_changed()

        computer_tab = ttk.Frame(self.source_tabs, padding=8)
        self.source_tabs.add(computer_tab, text="From this computer")
        self.local_path_var = tk.StringVar()
        ttk.Label(computer_tab, text="OBJ file").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self.local_path_entry = ttk.Entry(
            computer_tab, textvariable=self.local_path_var, state="readonly"
        )
        self.local_path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(
            computer_tab, text="Browse…", command=self._browse_obj
        ).grid(row=0, column=2)
        computer_tab.columnconfigure(1, weight=1)

        controls = ttk.Frame(selector)
        controls.grid(row=0, column=1, sticky="ns")
        ttk.Label(controls, text="Search method").grid(
            row=0, column=0, sticky="w"
        )
        self.search_method_var = tk.StringVar(value="Brute Force")
        self.search_method_combo = ttk.Combobox(
            controls,
            textvariable=self.search_method_var,
            values=("Brute Force", "KNN (cKDTree)", "RNN (cKDTree)"),
            state="readonly",
            width=18,
        )
        self.search_method_combo.grid(
            row=1, column=0, sticky="ew", pady=(3, 8)
        )
        self.search_method_combo.bind(
            "<<ComboboxSelected>>", self._search_method_changed
        )
        self.search_parameter_label = ttk.Label(controls, text="Top K")
        self.search_parameter_label.grid(row=2, column=0, sticky="w")
        self.top_k_var = tk.StringVar(value=str(top_k))
        self.top_k_input = ttk.Spinbox(
            controls,
            from_=1,
            to=1000000,
            textvariable=self.top_k_var,
            width=8,
        )
        self.top_k_input.grid(row=3, column=0, sticky="ew", pady=(3, 8))
        self.radius_var = tk.StringVar()
        self.radius_input = ttk.Entry(
            controls, textvariable=self.radius_var, width=12
        )
        self.radius_input.grid(row=3, column=0, sticky="ew", pady=(3, 8))
        self.radius_input.grid_remove()
        self.search_button = ttk.Button(
            controls, text="Find nearest meshes", command=self._start_search
        )
        self.search_button.grid(row=4, column=0, sticky="ew")

        self.status_var = tk.StringVar(
            value="Select a database mesh or browse for an OBJ file."
        )
        ttk.Label(
            self.root, textvariable=self.status_var, anchor="w"
        ).pack(fill="x", padx=12, pady=(0, 7))

        self.main_pane = ttk.Panedwindow(self.root, orient="horizontal")
        self.main_pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.query_panel = ttk.LabelFrame(
            self.main_pane, text="Query mesh", padding=8
        )
        self.results_panel = ttk.LabelFrame(
            self.main_pane, text="Nearest matches", padding=8
        )
        self.main_pane.add(self.query_panel, weight=1)
        self.main_pane.add(self.results_panel, weight=3)

        self.query_name_var = tk.StringVar(value="No query run yet")
        ttk.Label(
            self.query_panel,
            textvariable=self.query_name_var,
            wraplength=360,
            justify="left",
        ).pack(fill="x", pady=(0, 4))
        self.query_preview = ttk.Frame(self.query_panel, width=370, height=330)
        self.query_preview.pack(fill="both", expand=True)
        self.query_preview.pack_propagate(False)
        self.query_features_button = ttk.Button(
            self.query_panel,
            text="Query feature values…",
            command=self._show_query_feature_values,
            state="disabled",
        )
        self.query_features_button.pack(fill="x", pady=(8, 0))

        results_header = ttk.Frame(self.results_panel)
        results_header.pack(fill="x", pady=(0, 5))
        ttk.Label(
            results_header,
            text="Drag a mesh to rotate its 3D preview; scroll to browse results.",
        ).pack(side="left")

        list_frame = ttk.Frame(self.results_panel)
        list_frame.pack(fill="both", expand=True)
        self.results_canvas = self.tk.Canvas(
            list_frame, highlightthickness=0, borderwidth=0
        )
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self._scroll_results
        )
        self.results_canvas.configure(yscrollcommand=scrollbar.set)
        self.results_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.results_inner = ttk.Frame(self.results_canvas)
        self.results_window = self.results_canvas.create_window(
            (0, 0), window=self.results_inner, anchor="nw"
        )
        self.results_inner.bind(
            "<Configure>", self._results_inner_configured
        )
        self.results_canvas.bind(
            "<Configure>", self._results_canvas_configured
        )
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self.root.after(120, lambda: self.main_pane.sashpos(0, 410))

    def _search_method_changed(self, _event=None):
        radius_mode = self.search_method_var.get().startswith("RNN")
        self.search_parameter_label.configure(
            text="Radius (combined score)" if radius_mode else "Top K"
        )
        if radius_mode:
            self.top_k_input.grid_remove()
            self.radius_input.grid()
        else:
            self.radius_input.grid_remove()
            self.top_k_input.grid()

    def _category_changed(self, _event=None):
        category = self.category_var.get()
        filenames = [
            path.name for path in self.meshes_by_category.get(category, [])
        ]
        self.mesh_combo.configure(values=filenames)
        if filenames:
            self.mesh_combo.current(0)
        else:
            self.mesh_var.set("")

    def _browse_obj(self):
        from tkinter import filedialog

        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select a query OBJ mesh",
            filetypes=(("OBJ meshes", "*.obj"), ("All files", "*")),
        )
        if selected:
            self.local_path_var.set(selected)

    def _selected_mesh_path(self):
        if self.source_tabs.index("current") == 0:
            category = self.category_var.get()
            filename = self.mesh_var.get()
            for path in self.meshes_by_category.get(category, []):
                if path.name == filename:
                    return path
            return None

        path_text = self.local_path_var.get().strip()
        if not path_text:
            return None
        path = Path(path_text)
        if path.suffix.casefold() != ".obj":
            raise ValueError("The selected file must have an .obj extension.")
        if not path.is_file():
            raise FileNotFoundError(f"OBJ file not found: {path}")
        return path

    def _start_search(self):
        try:
            query_path = self._selected_mesh_path()
            if query_path is None:
                raise ValueError("Select a mesh before starting the search.")
            selected_method = self.search_method_var.get()
            search_method = {
                "Brute Force": "brute_force",
                "KNN (cKDTree)": "knn",
                "RNN (cKDTree)": "rnn",
            }[selected_method]
            radius = None
            top_k = 5
            if search_method == "rnn":
                radius = float(self.radius_var.get())
                if not np.isfinite(radius) or radius < 0:
                    raise ValueError(
                        "Radius must be a finite non-negative number."
                    )
            else:
                top_k = int(self.top_k_var.get())
                if top_k < 1:
                    raise ValueError("Top K must be at least one.")
        except (ValueError, FileNotFoundError) as error:
            self.messagebox.showerror(
                "Invalid query", str(error), parent=self.root
            )
            return

        self.search_button.state(["disabled"])
        self.status_var.set(
            f"Processing {query_path.name} and ranking database meshes…"
        )

        def run_query():
            try:
                result = self.query_database(
                    query_path,
                    self.database_path,
                    self.features_path,
                    source_path=self.source_path,
                    top_k=top_k,
                    seed=self.seed,
                    search_method=search_method,
                    radius=radius,
                    search_index=self.search_index,
                )
            except Exception as error:
                self.result_queue.put((False, error))
            else:
                self.result_queue.put((True, (query_path, result)))

        threading.Thread(target=run_query, daemon=True).start()
        self.root.after(100, self._check_search_result)

    def _check_search_result(self):
        if self._closed:
            return
        try:
            succeeded, result = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._check_search_result)
            return

        self.search_button.state(["!disabled"])
        if not succeeded:
            self.status_var.set("The query failed.")
            self.messagebox.showerror(
                "Query failed", str(result), parent=self.root
            )
            return

        query_path, (query_mesh, quality, matches) = result
        self._show_query(query_path, query_mesh, quality)
        self._show_results(quality, matches)

    def _show_query(self, query_path, query_mesh, quality):
        self.query_name_var.set(
            f"{query_path.parent.name}/{query_path.name}\n"
            f"Preprocessed preview"
        )
        old_figure = getattr(self, "query_figure", None)
        if old_figure is not None:
            old_figure.clear()
        self.query_figure, self.query_canvas = self._draw_mesh(
            self.query_preview, query_mesh, (3.4, 3.0)
        )
        self.query_feature_values = quality["Query_feature_values"]
        self.query_features_button.state(["!disabled"])

    def _show_results(self, quality, matches):
        self._clear_results()
        method_label = {
            "brute_force": "Brute force",
            "knn": "KNN (cKDTree)",
            "rnn": "RNN (cKDTree)",
        }[quality["Search_method_used"]]
        status = (
            f"{method_label}: showing {len(matches)} of "
            f"{quality['Candidates_ranked']:,} candidates "
            "(lower distance is better)."
        )
        if quality["Search_fallback_reason"]:
            status += (
                " Exact brute-force fallback: "
                f"{quality['Search_fallback_reason']}."
            )
        if quality["Search_method_used"] == "rnn":
            status += f" Radius: {quality['RNN_radius']:.8g}."
        self.status_var.set(status)
        if not matches:
            self.ttk.Label(
                self.results_inner, text="No matching candidates were returned."
            ).pack(anchor="w", padx=8, pady=8)
            return

        query_features = quality["Query_feature_values"]
        for rank, match in enumerate(matches, start=1):
            card = self.ttk.LabelFrame(
                self.results_inner,
                text=f"#{rank}  {match['Class']}/{match['File']}",
                padding=8,
            )
            card.pack(fill="x", expand=True, padx=4, pady=5)
            card.columnconfigure(1, weight=1)

            preview = self.ttk.Frame(card, width=330, height=230)
            preview.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 10))
            preview.grid_propagate(False)
            self._set_preview_placeholder(
                preview, "Scroll into view to load the 3D preview."
            )

            self.ttk.Label(
                card,
                text=f"Overall distance: {match['Score']:.8g}",
                font=("", 11, "bold"),
            ).grid(row=0, column=1, sticky="w", pady=(2, 8))
            unavailable = match["Unavailable_descriptors"]
            if unavailable:
                self.ttk.Label(
                    card,
                    text="Unavailable features: " + ", ".join(unavailable),
                    wraplength=600,
                ).grid(row=1, column=1, sticky="w", pady=(4, 6))
            self.ttk.Button(
                card,
                text="Interactive 3D view…",
                command=lambda item=match: self._show_interactive_result(item),
            ).grid(row=2, column=1, sticky="w", pady=(6, 2))
            self.ttk.Button(
                card,
                text="Feature values and distances…",
                command=lambda item=match, qf=query_features: (
                    self._show_feature_details(item, qf)
                ),
            ).grid(row=3, column=1, sticky="w", pady=(4, 2))
            self.result_cards.append({
                "card": card,
                "preview": preview,
                "match": match,
                "rendered": False,
            })

        self._schedule_preview_refresh()

    @staticmethod
    def _format_distance(value):
        return "unavailable" if value is None else f"{value:.8g}"

    def _clear_results(self):
        for card in self.result_cards:
            self._remove_preview(card)
        self.result_cards.clear()
        for child in self.results_inner.winfo_children():
            child.destroy()

    def _set_preview_placeholder(self, frame, text):
        for child in frame.winfo_children():
            child.destroy()
        self.ttk.Label(
            frame, text=text, wraplength=300, justify="center"
        ).pack(fill="both", expand=True)

    def _draw_mesh(self, frame, mesh, figure_size):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        for child in frame.winfo_children():
            child.destroy()
        figure = Figure(figsize=figure_size, dpi=80)
        axis = figure.add_subplot(111, projection="3d")
        self._add_mesh_to_axis(axis, mesh, linewidth=0.3)
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return figure, canvas

    @staticmethod
    def _add_mesh_to_axis(axis, mesh, linewidth):
        from matplotlib.colors import LightSource
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        bounds = np.asarray(mesh.bounds, dtype=float)
        center = bounds.mean(axis=0)
        radius = max(float(np.ptp(bounds, axis=0).max()) / 2.0, 1e-6)

        axis.add_collection3d(Poly3DCollection(
            vertices[faces],
            facecolors="#8db7d5",
            edgecolors="#263b4a",
            linewidth=linewidth,
            shade=True,
            lightsource=LightSource(azdeg=315, altdeg=45),
        ))

        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("persp")
        axis.view_init(elev=25, azim=-55)
        axis.set_axis_off()
        axis.set_facecolor("#f4f6f8")

    def _show_interactive_result(self, match):
        try:
            mesh = mmru.load_mesh(match["Path"])
        except Exception as error:
            self.messagebox.showerror(
                "Could not load result mesh", str(error), parent=self.root
            )
            return

        window = self.tk.Toplevel(self.root)
        window.title(f"{match['Class']}/{match['File']}")
        window.geometry("760x650")
        self.ttk.Label(
            window,
            text=f"Overall distance: {match['Score']:.8g}",
        ).pack(anchor="w", padx=10, pady=(8, 0))
        frame = self.ttk.Frame(window)
        frame.pack(fill="both", expand=True)
        window.mesh_figure, window.mesh_canvas = self._draw_mesh(
            frame, mesh, (7.2, 5.8)
        )

    def _render_card(self, card):
        try:
            mesh_path = Path(card["match"]["Path"])
            mesh = mmru.load_mesh(mesh_path)
            card["photo"] = self._render_preview_image(mesh)
            self._set_preview_image(card["preview"], card["photo"])
        except Exception as error:
            self._set_preview_placeholder(
                card["preview"], f"Could not display mesh:\n{error}"
            )
        card["rendered"] = True

    def _render_preview_image(self, mesh):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(3.2, 2.2), dpi=80)
        axis = figure.add_subplot(111, projection="3d")
        self._add_mesh_to_axis(axis, mesh, linewidth=0.22)
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        image_buffer = io.BytesIO()
        canvas.print_png(image_buffer)
        figure.clear()
        return self.tk.PhotoImage(
            data=base64.b64encode(image_buffer.getvalue()).decode("ascii")
        )

    def _set_preview_image(self, frame, image):
        for child in frame.winfo_children():
            child.destroy()
        self.ttk.Label(frame, image=image).pack(fill="both", expand=True)

    def _remove_preview(self, card):
        self.preview_cache.pop(id(card), None)
        card.pop("photo", None)
        if card.get("rendered") and card["preview"].winfo_exists():
            self._set_preview_placeholder(
                card["preview"], "Scroll into view to load the 3D preview."
            )
        card["rendered"] = False

    def _schedule_preview_refresh(self):
        if self._closed:
            return
        if self._preview_refresh_after is not None:
            self.root.after_cancel(self._preview_refresh_after)
        self._preview_refresh_after = self.root.after(
            140, self._refresh_visible_previews
        )

    def _refresh_visible_previews(self):
        self._preview_refresh_after = None
        if self._closed:
            return
        top = self.results_canvas.canvasy(0) - 80
        bottom = (
            self.results_canvas.canvasy(self.results_canvas.winfo_height()) + 80
        )
        visible_ids = set()
        for card in self.result_cards:
            y0 = card["card"].winfo_y()
            y1 = y0 + card["card"].winfo_height()
            if y1 >= top and y0 <= bottom:
                visible_ids.add(id(card))
                if not card["rendered"]:
                    self._render_card(card)
                if "figure" in card:
                    self.preview_cache.pop(id(card), None)
                    self.preview_cache[id(card)] = card

        while len(self.preview_cache) > MAX_CACHED_PREVIEWS:
            evicted = next(
                (
                    key for key in self.preview_cache
                    if key not in visible_ids
                ),
                None,
            )
            if evicted is None:
                break
            self._remove_preview(self.preview_cache[evicted])

    def _show_feature_details(self, match, query_features):
        columns = (
            ("query_value", "Query value", 210),
            ("candidate_value", "Mesh value", 210),
            ("distance", "Distance", 140),
            ("weight", "Weight", 100),
            ("weighted", "Weighted contribution", 170),
        )
        tree = self._create_feature_tree(
            f"Feature comparison: {match['Class']}/{match['File']}",
            "Scalar distances use standardized values. Histogram distances "
            "use Wasserstein distance; expand a histogram to inspect its bins.",
            columns,
            "1180x650",
        )
        candidate_features = match["Feature_values"]
        distances = match["Distances"]
        weighted = match["Weighted_distances"]
        for name, weight in self.descriptor_weights.items():
            query_value = query_features.get(name)
            candidate_value = candidate_features.get(name)
            if isinstance(query_value, np.ndarray) or isinstance(
                candidate_value, np.ndarray
            ):
                distance = self._format_distance(distances.get(name))
                contribution = self._format_distance(weighted.get(name))
                parent = tree.insert(
                    "",
                    "end",
                    text=name,
                    values=(
                        "25-bin normalized histogram"
                        if isinstance(query_value, np.ndarray) else "unavailable",
                        "25-bin normalized histogram"
                        if isinstance(candidate_value, np.ndarray) else "unavailable",
                        distance,
                        f"{weight:g}",
                        contribution,
                    ),
                )
                query_bins = (
                    query_value if isinstance(query_value, np.ndarray) else []
                )
                candidate_bins = (
                    candidate_value
                    if isinstance(candidate_value, np.ndarray) else []
                )
                for index in range(max(len(query_bins), len(candidate_bins))):
                    query_bin = (
                        self._format_number(query_bins[index])
                        if index < len(query_bins) else "unavailable"
                    )
                    candidate_bin = (
                        self._format_number(candidate_bins[index])
                        if index < len(candidate_bins) else "unavailable"
                    )
                    tree.insert(
                        parent,
                        "end",
                        text=f"Bin {index}",
                        values=(query_bin, candidate_bin, "", "", ""),
                    )
            else:
                tree.insert(
                    "",
                    "end",
                    text=name,
                    values=(
                        self._format_number(query_value),
                        self._format_number(candidate_value),
                        self._format_distance(distances.get(name)),
                        f"{weight:g}",
                        self._format_distance(weighted.get(name)),
                    ),
                )

    def _show_query_feature_values(self):
        query_features = getattr(self, "query_feature_values", {})
        columns = (("value", "Query value", 260),)
        tree = self._create_feature_tree(
            "Query feature values",
            "Scalar values are raw measurements. Expand a histogram to inspect "
            "its normalized bin proportions.",
            columns,
            "600x650",
        )
        for name in self.descriptor_weights:
            value = query_features.get(name)
            if isinstance(value, np.ndarray):
                parent = tree.insert(
                    "",
                    "end",
                    text=name,
                    values=("25-bin normalized histogram",),
                )
                for index, bin_value in enumerate(value):
                    tree.insert(
                        parent,
                        "end",
                        text=f"Bin {index}",
                        values=(self._format_number(bin_value),),
                    )
            else:
                tree.insert(
                    "",
                    "end",
                    text=name,
                    values=(self._format_number(value),),
                )

    def _create_feature_tree(self, title, description, columns, geometry):
        tk, ttk = self.tk, self.ttk
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry(geometry)
        ttk.Label(
            window, text=description, wraplength=1100, justify="left"
        ).pack(anchor="w", padx=10, pady=8)

        frame = ttk.Frame(window, padding=(10, 0, 10, 10))
        frame.pack(fill="both", expand=True)
        data_columns = tuple(column[0] for column in columns)
        tree = ttk.Treeview(
            frame, columns=data_columns, show=("tree", "headings")
        )
        tree.heading("#0", text="Feature / bin")
        tree.column("#0", width=210, anchor="w")
        for column, heading, width in columns:
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="w")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(
            frame, orient="horizontal", command=tree.xview
        )
        tree.configure(
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set,
        )
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    @staticmethod
    def _format_number(value):
        if value is None:
            return "unavailable"
        try:
            return f"{float(value):.8g}" if np.isfinite(value) else "unavailable"
        except (TypeError, ValueError):
            return str(value)

    def _results_inner_configured(self, _event=None):
        self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox("all")
        )
        self._schedule_preview_refresh()

    def _results_canvas_configured(self, event):
        self.results_canvas.itemconfigure(
            self.results_window, width=event.width
        )
        self._schedule_preview_refresh()

    def _scroll_results(self, *args):
        self.results_canvas.yview(*args)
        self._schedule_preview_refresh()

    def _on_mousewheel(self, event):
        if str(event.widget).startswith(str(self.results_canvas)):
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            elif abs(event.delta) < 120 and event.delta:
                delta = -1 if event.delta > 0 else 1
            else:
                delta = -int(event.delta / 120) if event.delta else 0
            if delta:
                self.results_canvas.yview_scroll(delta, "units")
                self._schedule_preview_refresh()
            return "break"
        return None

    def _close(self):
        self._closed = True
        if self.root is not None:
            if self._preview_refresh_after is not None:
                self.root.after_cancel(self._preview_refresh_after)
            self.root.destroy()

    def run(self):
        if self.root is not None:
            self.root.mainloop()


def launch_query_ui(
    query_database,
    descriptor_weights,
    database_path,
    source_path,
    features_path,
    top_k=5,
    seed=None,
    search_index=None,
):
    """Open the query picker and interactive ranked-results window."""
    app = QueryWindow(
        query_database,
        descriptor_weights,
        database_path,
        source_path,
        features_path,
        top_k=top_k,
        seed=seed,
        search_index=search_index,
    )
    app.run()
