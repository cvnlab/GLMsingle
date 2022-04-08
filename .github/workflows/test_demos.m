function test_suite = test_demos %#ok<*STOUT>

  try % assignment of 'localfunctions' is necessary in Matlab >= 2016
    test_functions = localfunctions(); %#ok<*NASGU>
  catch % no problem; early Matlab versions can use initTestSuite fine
  end

  initTestSuite;

end

function test_demo1()

  run(fullfile(root_dir(), 'matlab', 'examples', 'example1'));

end

function test_demo2()

  run(fullfile(root_dir(), 'matlab', 'examples', 'example2'));

end

function value = root_dir()

  value = getenv('GITHUB_WORKSPACE');

  if isempty(value)
    value =  fullfile(fileparts(mfilename('fullpath')), '..', '..');
  end

end
