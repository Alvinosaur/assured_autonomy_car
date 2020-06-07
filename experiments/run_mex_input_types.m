some_matrix = magic(3);
disp("input mat:");
disp(some_matrix);

some_vec = [39, 122000];  % 1 x 2
disp("input vec:");
disp(some_vec);

output = mex_input_types(some_matrix, some_vec);

disp("output:");
disp(output);