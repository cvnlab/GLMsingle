% run demos with moxunit in github CI

root_dir = getenv('GITHUB_WORKSPACE');

% MOxUnit and MOcov need to be in the matlab path
addpath(fullfile(root_dir, 'MOcov', 'MOcov'));
cd(fullfile(root_dir, 'MOxUnit', 'MOxUnit'));
run moxunit_set_path();

% add glm single to path
cd(root_dir);
setup();

this_folder = fileparts(mfilename('fullpath'));
test_folder = this_folder;
success = moxunit_runtests(test_folder, '-verbose', '-recursive');

if success
  system('echo 0 > test_report.log');
else
  system('echo 1 > test_report.log');
end
