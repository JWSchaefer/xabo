def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-fixtures",
        action="store_true",
        default=False,
        help="Regenerate test fixtures",
    )
