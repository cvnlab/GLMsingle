root_dir = getenv('GITHUB_WORKSPACE');

% MOxUnit and MOcov need to be in the matlab path
addpath(fullfile(root_dir, 'MOcov', 'MOcov'));
cd(fullfile(root_dir, 'MOxUnit', 'MOxUnit'));
run moxunit_set_path();

% adds GLM single to the path and runs all the tests
cd(fullfile(root_dir));
setup();
run run_tests();
