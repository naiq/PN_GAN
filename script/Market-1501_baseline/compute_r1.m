function r1 = compute_r1( query_files, test_files, k, index )
    r1 = 0;
    count = 1;
    rank_id = str2double(test_files(index(count)).name(1:4)); %the same id,the same person
    rank_cam = test_files(index(count)).name(7);  % camera number
    query_id = str2double(query_files(k).name(1:4));
    query_cam = query_files(k).name(7);
    while( rank_id == query_id )
        if( rank_cam == query_cam) % ignore the same person from the same camera
            count = count+1;
            rank_id = str2double(test_files(index(count)).name(1:4));
            rank_cam = test_files(index(count)).name(7);
        elseif( rank_cam ~= query_cam) % hit!
            r1 = r1 + 1;
            break;
        end
    end
end

