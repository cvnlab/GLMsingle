function run_tests()

  tic;

  cd(fileparts(mfilename('fullpath')));

  fprintf('\nHome is %s\n', getenv('HOME'));

  folder_to_cover = fullfile(pwd, 'matlab');

  test_folder = fullfile(pwd, 'tests');

  success = moxunit_runtests(test_folder, ...
                             '-verbose', '-recursive', '-with_coverage', ...
                             '-cover', folder_to_cover, ...
                             '-cover_xml_file', 'coverage.xml', ...
                             '-cover_html_dir', fullfile(pwd, 'coverage_html'));

  if success
    system('echo 0 > test_report.log');
  else
    system('echo 1 > test_report.log');
  end

  toc;

end
