from solo_odds.units import parse_hashrate, format_hashrate

def test_parse_hashrate_th() -> None:
    hr = parse_hashrate("9.4TH")
    assert int(hr.hs) == int(9.4e12)

def test_parse_hashrate_with_spaces() -> None:
    hr = parse_hashrate("1200 GH/s")
    assert int(hr.hs) == int(1.2e12)

def test_format_hashrate() -> None:
    assert format_hashrate(9.4e12).endswith("TH/s")