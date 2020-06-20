is_transposed = false;
some_matrix = [
    1,2,3;
    4,5,6];
if is_transposed
    some_matrix = some_matrix';
end
disp("input mat:");
disp(some_matrix);

some_vec = [39, 122000];  % 1 x 2
disp("input vec:");
disp(some_vec);

output = mex_input_types(some_matrix, some_vec, is_transposed);

disp("output:");
disp(output);