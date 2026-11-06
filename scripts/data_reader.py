import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
SKIP_DIR_NAMES = frozenset({"raw"})


def ensure_directory(path):
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError("directory does not exist: %s" % folder)
    return folder.resolve()


def resolve_dataset_path(dataset_name):
    """Resolve a path under DATASET_ROOT from a dataset subfolder name.

    Examples: ``nhtsa_2022``, ``openaerialmap``.
    """
    root = DATASET_ROOT.resolve()
    rel = Path(str(dataset_name).strip("/\\"))
    if rel.parts and rel.parts[0] == "dataset":
        rel = Path(*rel.parts[1:])
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("dataset path must stay under %s: %s" % (root, dataset_name))
    return ensure_directory(path)


def list_files_in_folder(folder_path):
    folder = ensure_directory(folder_path)
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


def resolve_file_in_folder(folder_path, filename):
    path = ensure_directory(folder_path) / filename
    if path.is_file():
        return path.resolve()
    raise FileNotFoundError("file not found: %s" % path)


def read_file_bytes(path):
    return Path(path).read_bytes()


def random_choice(items, seed):
    pool = list(items)
    if not pool:
        raise ValueError("cannot choose from an empty sequence")
    return random.Random(seed).choice(pool)


def find_image_in_folder(folder_path):
    folder = ensure_directory(folder_path)
    for pattern in ("*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        images = sorted(folder.glob(pattern))
        if images:
            return images[0].resolve()
    raise FileNotFoundError("no image under %s" % folder)


def pdf_to_png(pdf_path, out_png, dpi=300, page_no=0):
    import io

    import fitz
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_no)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    finally:
        doc.close()

    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    lut = [int((i / 255.0) ** 2.0 * 255) for i in range(256)]
    g = g.point(lut)
    g = ImageEnhance.Contrast(g).enhance(1.6)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=2))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    g.save(str(out_png))


class DatasetReader(object):
    """Read images from ``dataset/<dataset_name>/<case_id>/`` (or flat files)."""

    def __init__(self, dataset_name):
        self.dataset_name = str(dataset_name).strip("/\\")
        self.folder_path = resolve_dataset_path(self.dataset_name)
        self.case_id = None
        self.filename = None
        self.file_path = None
        self.png_path = None
        self.data = None

    def list_cases(self):
        """Return case folder names under the dataset that contain an image."""
        cases = []
        for path in sorted(self.folder_path.iterdir(), key=lambda p: p.name):
            if not path.is_dir() or path.name.startswith(".") or path.name in SKIP_DIR_NAMES:
                continue
            try:
                find_image_in_folder(path)
            except FileNotFoundError:
                continue
            cases.append(path.name)
        return cases

    def read(self, name):
        """Load by case folder name, or by filename directly under the dataset root."""
        name = str(name)
        case_dir = self.folder_path / name
        if case_dir.is_dir():
            path = find_image_in_folder(case_dir)
            self.case_id = name
            return self._set(path)
        path = resolve_file_in_folder(self.folder_path, name)
        self.case_id = None
        return self._set(path)

    def read_random(self, seed):
        """Load a random case (or flat file if there are no case folders)."""
        cases = self.list_cases()
        if cases:
            return self.read(random_choice(cases, seed))
        files = list_files_in_folder(self.folder_path)
        if not files:
            raise FileNotFoundError("no cases or files under %s" % self.folder_path)
        path = random_choice(files, seed)
        self.case_id = None
        return self._set(path)

    def _set(self, path):
        path = Path(path).resolve()
        self.filename = path.name
        self.file_path = path
        self.png_path = path
        self.data = read_file_bytes(path)
        return self
