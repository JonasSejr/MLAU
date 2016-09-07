src = fullfile(pwd, '/src')
addpath(src)
import matlab.unittest.TestSuite;
run(TestSuite.fromFolder(fullfile(pwd, '/tests')));