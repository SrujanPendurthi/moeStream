def test_lab_serve_loads_the_given_pack_under_the_model_name_pi_sends(monkeypatch):
    import sys

    import scripts.serve_streaming as lab

    import tools.lab_serve as ls

    seen = {}
    monkeypatch.setattr(lab, "main", lambda: seen.update(argv=list(sys.argv), pack=lab.pack))
    monkeypatch.setattr(sys, "argv", ["x"])
    ls.main(["--pack", "/abs/pack", "--slots", "64", "--port", "8081"])
    assert seen["pack"] == "/abs/pack"
    a = seen["argv"]
    assert a[a.index("--model") + 1] == "Laguna-XS-2.1"       # no reload when pi names this model
    assert a[a.index("--slots") + 1] == "64" and "--port" in a
    assert a[a.index("--max-tokens") + 1] == "4096"
