root_dir = getenv('GITHUB_WORKSPACE');

addpath(fullfile(root_dir, 'MOcov', 'MOcov'));

cd(fullfile(root_dir, 'MOxUnit', 'MOxUnit'));
run moxunit_set_path();

cd(fullfile(root_dir));

setup();

run run_tests();
