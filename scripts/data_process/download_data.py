import time
import random
import base64
import re
import zipfile
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ============================================================
# 配置
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_OUT_DIR = (REPO_ROOT / "dataset" / "nhtsa_raw").resolve()
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAGES = 5
MAX_CASES = 50


# ============================================================
# PDF 保存与校验
# ============================================================
MIN_PDF_BYTES = 3 * 1024


def is_pdf_valid(path, min_bytes=MIN_PDF_BYTES):
    if not path.exists():
        return False
    try:
        size = path.stat().st_size
        if size < min_bytes:
            return False
        header = path.read_bytes()[:4]
        return header == b"%PDF"
    except Exception:
        return False


def save_as_pdf(page, output_path, min_bytes=MIN_PDF_BYTES):
    try:
        client = page.context.new_cdp_session(page)
        result = client.send("Page.printToPDF", {
            "landscape": False,
            "displayHeaderFooter": False,
            "printBackground": True,
            "paperWidth": 8.27,
            "paperHeight": 11.69,
            "marginTop": 0.4,
            "marginBottom": 0.4,
            "marginLeft": 0.4,
            "marginRight": 0.4
        })
        pdf_data = base64.b64decode(result["data"])
        if len(pdf_data) < min_bytes or b"%PDF" not in pdf_data[:1024]:
            print("   ⚠️ 生成到的 PDF 数据异常，准备重试...")
            return False
        output_path.write_bytes(pdf_data)
        return True
    except Exception as e:
        print(f"   ❌ PDF 生成失败: {e}")
        return False


def wait_locator_visible(locator, retries=3, delay=1.0):
    for _ in range(retries):
        try:
            if locator.is_visible():
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


# ============================================================
# 网页解析工具（修复 strict mode）
# ============================================================
def extract_case_year_from_text(text, cid=None):
    patterns = []
    if cid:
        patterns.append(rf"\b{re.escape(cid)}[_-].*?(\d{{4}})\b")
    patterns.extend([
        r"\b\d{1,2}[-/]\d{1,2}[-/](\d{4})[-/]\d{3}[-/]\d{2}\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/](\d{4})\b",
    ])
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)
    return None


def resolve_case_year(page, cid):
    texts = []
    try:
        texts.append(page.locator("body").inner_text(timeout=30000))
    except Exception:
        pass
    try:
        texts.append(page.content())
    except Exception:
        pass
    try:
        texts.append(page.title())
    except Exception:
        pass

    year = None
    for t in texts:
        year = extract_case_year_from_text(t, cid=cid)
        if year:
            break
    if not year:
        raise RuntimeError(f"无法从页面解析 case 年份: {cid}")
    return year


# ============================================================
# 主流程
# ============================================================
def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )

        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            accept_downloads=True
        )

        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        context.add_init_script("window.print = function(){};")

        page = context.new_page()
        page.goto("https://crashviewer.nhtsa.dot.gov/ciss/searchindex", wait_until="domcontentloaded")

        print("\n====================================================")
        print("👋 请人工点击 SEARCH，看到表格后按 Enter")
        print("====================================================\n")
        print(f"📁 输出目录: {BASE_OUT_DIR}（按 case 年份分目录）")
        input("⌨️ 回车继续: ")

        # --------------------------------------------------
        # 扫描案件（支持分页与总数限制）
        # --------------------------------------------------
        tasks = []
        seen = set()

        def goto_next_page():
            selectors = [
                "button[aria-label='Next']",
                "a[aria-label='Next']",
                "button:has-text('Next')",
                "a:has-text('Next')",
                "button:has-text('›')",
                "a:has-text('›')",
                "button:has-text('>')",
                "a:has-text('>')",
                "text=/^\\s*Next\\s*$/",
                "text=/^\\s*>\\s*$/",
            ]
            for sel in selectors:
                loc = page.locator(sel).first
                try:
                    if loc.count() == 0 or not loc.is_visible():
                        continue
                    if not loc.is_enabled():
                        continue
                    aria_disabled = (loc.get_attribute("aria-disabled") or "").lower()
                    disabled = (loc.get_attribute("disabled") or "").lower()
                    cls = (loc.get_attribute("class") or "").lower()
                    if aria_disabled in ("true", "1") or disabled in ("true", "disabled", "1"):
                        continue
                    if "disabled" in cls:
                        continue
                    prev_marker = None
                    try:
                        prev_marker = page.locator("tbody tr").first.inner_text(timeout=30000)
                    except Exception:
                        pass
                    loc.click()
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=60000)
                    except Exception:
                        pass
                    if prev_marker:
                        try:
                            page.wait_for_function(
                                "(prev) => { const row = document.querySelector('tbody tr'); return row && row.innerText.trim() !== prev; }",
                                arg=prev_marker,
                                timeout=10000,
                            )
                        except Exception:
                            pass
                    return True
                except Exception:
                    continue
            return False

        page.locator("tbody tr td").first.wait_for(state="visible", timeout=60000)
        pages_scanned = 0

        while pages_scanned < MAX_PAGES and len(tasks) < MAX_CASES:
            rows = page.locator("tbody tr")
            for idx in range(rows.count()):
                if len(tasks) >= MAX_CASES:
                    break
                row = rows.nth(idx)
                tds = row.locator("td")
                if tds.count() >= 2:
                    link = tds.nth(0).locator("a").first
                    cid = tds.nth(1).inner_text().strip()
                    if cid.isdigit() and link.count() > 0 and cid not in seen:
                        case_name = link.inner_text().strip()
                        case_year = extract_case_year_from_text(case_name, cid=cid)
                        href = link.get_attribute("href")
                        if href:
                            seen.add(cid)
                            tasks.append({
                                "id": cid,
                                "year": case_year,
                                "url": "https://crashviewer.nhtsa.dot.gov" + href
                            })

            pages_scanned += 1
            if len(tasks) >= MAX_CASES or pages_scanned >= MAX_PAGES:
                break
            if not goto_next_page():
                break

        print(f"✅ 实际处理 {len(tasks)} 个 case（MAX_CASES={MAX_CASES}, MAX_PAGES={MAX_PAGES}, 扫描页数={pages_scanned}）")

        # --------------------------------------------------
        # 逐案处理
        # --------------------------------------------------
        for i, item in enumerate(tasks, 1):
            cid = item["id"]
            url = item["url"]

            print(f"\n[{i}/{len(tasks)}] 🚀 处理 Case {cid}")

            case_year = item.get("year")
            if case_year:
                case_out_dir = BASE_OUT_DIR / case_year
                case_case_dir = case_out_dir / "cases"
                case_pdf_dir = case_out_dir / "pdfs"
                pdf_path = case_pdf_dir / f"{cid}_report.pdf"
                case_extract_dir = case_case_dir / cid
                has_case_files = (
                    case_extract_dir.exists() and any(case_extract_dir.iterdir())
                ) or bool(list(case_case_dir.glob(f"{cid}_*")))
                if is_pdf_valid(pdf_path) and has_case_files:
                    print("   ✅ PDF 和 Case Files 已存在，跳过该记录。")
                    continue

            dp = context.new_page()
            try:
                loaded = False
                for attempt in range(2):
                    try:
                        dp.goto(url, wait_until="domcontentloaded", timeout=600000)
                        loaded = True
                        break
                    except PlaywrightTimeoutError:
                        print(f"   ⚠️ 打开页面超时（第 {attempt + 1} 次），将重试...")
                        time.sleep(2.0)
                if not loaded:
                    print("   ❌ 页面打开失败，跳过该 case。")
                    continue
                time.sleep(2.5)

                case_year = case_year or resolve_case_year(dp, cid)
                case_out_dir = BASE_OUT_DIR / case_year
                case_case_dir = case_out_dir / "cases"
                case_pdf_dir = case_out_dir / "pdfs"
                for d in [case_out_dir, case_case_dir, case_pdf_dir]:
                    d.mkdir(parents=True, exist_ok=True)

                # ---------------- PDF ----------------
                pdf_path = case_pdf_dir / f"{cid}_report.pdf"
                if not is_pdf_valid(pdf_path):
                    print("   🖨️ 生成 PDF...")
                    btn = dp.locator("text=/Print Case|Print/").first
                    if wait_locator_visible(btn, retries=3, delay=1.0):
                        with context.expect_page() as pinfo:
                            btn.click()
                        pp = pinfo.value
                        # 采用更稳妥的等待策略：先等 load，再短暂停留
                        try:
                            pp.wait_for_load_state("load", timeout=60000)
                        except Exception as e:
                            # 即使 load 超时，也继续尝试打印，避免无结果
                            print(f"   ⚠️ 等待打印页加载超时，将继续尝试保存: {e}")
                        time.sleep(2.5)

                        # 多次尝试保存，确保尽量拿到结果
                        saved = False
                        for _ in range(3):
                            if save_as_pdf(pp, pdf_path) and is_pdf_valid(pdf_path):
                                saved = True
                                break
                            time.sleep(1.5)
                        if not saved:
                            raise RuntimeError("PDF 多次生成失败")
                        pp.close()

                # ---------------- CASE FILES ----------------
                case_extract_dir = case_case_dir / cid
                has_case_files = (case_extract_dir.exists() and any(case_extract_dir.iterdir())) or bool(
                    list(case_case_dir.glob(f"{cid}_*"))
                )
                if not has_case_files:
                    print("   ⬇️ 下载 Case Files...")
                    btn = dp.locator("text=/Download Case Files|Download/").first
                    if wait_locator_visible(btn, retries=3, delay=1.0):
                        saved = False
                        for attempt in range(3):
                            try:
                                with dp.expect_download(timeout=600000) as dinfo:
                                    btn.click(force=True)
                                    try:
                                        dp.locator("text=Export XML").first.click(force=True)
                                    except Exception:
                                        pass
                                d = dinfo.value
                                zip_path = case_case_dir / f"{cid}_{d.suggested_filename}"
                                d.save_as(zip_path)
                                if not zipfile.is_zipfile(zip_path):
                                    print(f"   ⚠️ 下载文件不是 zip（第 {attempt + 1} 次），将重试: {zip_path.name}")
                                    zip_path.unlink(missing_ok=True)
                                    time.sleep(1.0)
                                    continue
                                else:
                                    tmp_extract_dir = case_case_dir / f".tmp_extract_{cid}"
                                    tmp_extract_dir.mkdir(parents=True, exist_ok=True)
                                    try:
                                        with zipfile.ZipFile(zip_path, "r") as zf:
                                            zf.extractall(tmp_extract_dir)
                                        zip_path.unlink()

                                        extracted_items = list(tmp_extract_dir.iterdir())
                                        if not extracted_items:
                                            print(f"   ⚠️ 解压后为空（第 {attempt + 1} 次），将重试: {zip_path.name}")
                                            time.sleep(1.0)
                                            continue
                                        prefix = cid
                                        for p in extracted_items:
                                            if "_" in p.name:
                                                prefix = p.name.split("_", 1)[0] or cid
                                                break

                                        target_dir = case_case_dir / prefix
                                        target_dir.mkdir(parents=True, exist_ok=True)
                                        for p in extracted_items:
                                            shutil.move(str(p), target_dir)
                                        saved = True
                                    finally:
                                        if tmp_extract_dir.exists():
                                            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
                                if saved:
                                    break
                            except PlaywrightTimeoutError:
                                print(f"   ⚠️ 下载超时（第 {attempt + 1} 次），将重试...")
                                time.sleep(2.0)
                        if not saved:
                            print("   ⚠️ 下载失败，跳过该 case files。")
                    else:
                        print("   ⚠️ 未找到可见的 Download 按钮，跳过该 case files。")

            finally:
                dp.close()
                time.sleep(random.uniform(1.5, 3.0))

        browser.close()
        print("\n🎉 全部完成")


if __name__ == "__main__":
    main()
