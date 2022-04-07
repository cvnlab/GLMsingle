function test_suite = test_glmsingleunit %#ok<*STOUT>

  try % assignment of 'localfunctions' is necessary in Matlab >= 2016
    test_functions = localfunctions(); %#ok<*NASGU>
  catch % no problem; early Matlab versions can use initTestSuite fine
  end

  initTestSuite;

end

function test_glmsingleunit_smoke()

  [data, output_dir] = set_up();

  % GIVEN
  design = data.design(1:3);
  stimdur = data.stimdur;
  tr = data.tr;
  data = cellfun(@(x) x(51:70, 8:27, 1, :), data.data(1:3), 'UniformOutput', 0);

  results = GLMestimatesingletrial(design, ...
                                   data, ...
                                   stimdur, ...
                                   tr, ...
                                   output_dir, ...
                                   struct('wantmemoryoutputs', [1 1 1 1]));

  clean_up();

end

function [data, output_dir] = set_up()

  test_dir = fileparts(mfilename('fullpath'));

  data_dir = fullfile(test_dir, 'data');
  data_file = fullfile(data_dir, 'nsdcoreexampledataset.mat');

  output_dir = fullfile(test_dir, 'outputs', 'matlab');

  data = load(data_file);

  run(fullfile(test_dir, '..', 'setup.m'));

end

function clean_up()

end
