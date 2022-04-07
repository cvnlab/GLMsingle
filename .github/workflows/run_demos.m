root_dir = getenv('GITHUB_WORKSPACE');

cd(fullfile(root_dir));

setup();

run matlab/examples/example1;
run matlab/examples/example2;