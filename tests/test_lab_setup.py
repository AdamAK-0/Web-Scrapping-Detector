from pathlib import Path

from wsd.lab_setup import check_website_links, render_nginx_windows_conf, write_lab_artifacts


def test_website_links_are_resolvable():
    project_root = Path(__file__).resolve().parents[1]
    report = check_website_links(project_root)
    assert report["page_count"] >= 10
    assert report["missing_links"] == []
    assert report["missing_assets"] == []


def test_render_nginx_windows_conf_points_to_project_paths(tmp_path):
    conf = render_nginx_windows_conf(tmp_path, port=9090)
    assert "listen       9090;" in conf
    assert "website_lab" in conf
    assert "logs/access.log" in conf


def test_render_nginx_windows_conf_normalizes_explicit_log_paths(tmp_path):
    conf = render_nginx_windows_conf(tmp_path, access_log=r"C:\nginx\logs\access.log", error_log=r"C:\nginx\logs\error.log")
    assert "access_log  C:/nginx/logs/access.log  combined_custom;" in conf
    assert "error_log   C:/nginx/logs/error.log warn;" in conf


def test_write_lab_artifacts_creates_files(tmp_path):
    project = tmp_path / "project"
    (project / "website_lab").mkdir(parents=True)
    (project / "website_lab" / "index.html").write_text('<html><body><a href="about.html">About</a></body></html>', encoding='utf-8')
    (project / "website_lab" / "about.html").write_text('<html><body>About</body></html>', encoding='utf-8')
    paths = write_lab_artifacts(project, port=8081)
    assert Path(paths["conf_path"]).exists()
    assert Path(paths["report_path"]).exists()
