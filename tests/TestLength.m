classdef TestLength < matlab.unittest.TestCase
        
    methods (Test)
        function test1(tc)
            a = AClassToTest();
            tc.assertEqual(a.AddTwoNumbers(100, 100), 200);
        end
            
    end
     
end

