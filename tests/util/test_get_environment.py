from src.util.get_environment import get_datadir


class TestGetDataDir:
    def test_kaggleInteractive(self, mocker):
        mocker.patch('src.util.get_environment.get_exec_env', return_value='kaggle-Interactive')
        assert(get_datadir() == '/kaggle/working/data')

    def test_kaggleBatch(self, mocker):
        mocker.patch('src.util.get_environment.get_exec_env', return_value='kaggle-Batch')
        assert(get_datadir() == '/kaggle/working/data')

    def test_colab(self, mocker):
        mocker.patch('src.util.get_environment.get_exec_env', return_value='colab')
        assert(get_datadir() is None)

    def test_local(self, mocker):
        mocker.patch('src.util.get_environment.get_exec_env', return_value='local')
        assert(get_datadir() == '../data')
