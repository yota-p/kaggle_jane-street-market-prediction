from src import ex001_fe_download_raw


@param
ENV=
class TestEx001:
    def test_kaggle(self, mocker, tmpdir):
        EXNO = '001'
        DATA_DIR = str(tmpdir)
        IN_DIR = f'{DATA_DIR}/002'
        OUT_DIR = f'{DATA_DIR}/{EXNO}'
        Path(OUT_DIR).mkdir(exist_ok=True)

        DATA_DIR = str(tmpdir)
        mocker.patch('src.util.get_environment.get_datadir', return_value=DATA_DIR)
        mocker.patch('src.util.get_environment.get_exec_env', return_value=ENV)
        ex001_fe_download_raw.main()
        assert True
