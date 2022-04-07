function test_suite = test_GLMestimatesingletrial %#ok<*STOUT>

  try % assignment of 'localfunctions' is necessary in Matlab >= 2016
    test_functions = localfunctions(); %#ok<*NASGU>
  catch % no problem; early Matlab versions can use initTestSuite fine
  end

  initTestSuite;

end

function test_GLMestimatesingletrial_system()

  % "end-to-end" test of GLMestimatesingletrial
  % only checks the HRF index of 400 voxels

  [data, expected, output_dir] = set_up_test();

  % GIVEN
  design = data.design(1:3);
  stimdur = data.stimdur;
  tr = data.tr;
  data = cellfun(@(x) x(51:70, 8:27, 1, :), data.data(1:3), 'UniformOutput', 0);

  % WHEN
  results = GLMestimatesingletrial(design, ...
                                   data, ...
                                   stimdur, ...
                                   tr, ...
                                   output_dir, ...
                                   struct('wantmemoryoutputs', [1 1 1 1]));

  % THEN
  assertEqual(results{2}.HRFindex, expected{2}.HRFindex);
  assertEqual(results{3}.HRFindex, expected{3}.HRFindex);
  assertEqual(results{4}.HRFindex, expected{4}.HRFindex);

  assertElementsAlmostEqual(results{4}.R2, expected{4}.R2, 'absolute', 1e-1);

  clean_up();

end

function [data, expected, output_dir] = set_up_test()

  test_dir = fileparts(mfilename('fullpath'));

  data_dir = fullfile(test_dir, 'data');
  data_file = fullfile(data_dir, 'nsdcoreexampledataset.mat');
  data = load(data_file);

  expected_dir = fullfile(test_dir, 'expected', 'matlab');
  load(fullfile(expected_dir, 'TYPEB_FITHRF.mat'));
  expected{2}.HRFindex = HRFindex;
  load(fullfile(expected_dir, 'TYPEC_FITHRF_GLMDENOISE.mat'));
  expected{3}.HRFindex = HRFindex;
  load(fullfile(expected_dir, 'TYPED_FITHRF_GLMDENOISE_RR.mat'));
  expected{4}.HRFindex = HRFindex;
  expected{4}.R2 = R2;

  output_dir = fullfile(test_dir, 'outputs', 'matlab');

  run(fullfile(test_dir, '..', 'setup.m'));

end

function clean_up()

  % ununsed for now

end
