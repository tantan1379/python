#include <iostream>
#include <string.h>
#include <vector>

using namespace std;

class Solution {
public:
	//动态规划法
	int maxSubArray(vector<int>& nums) {
		int len = nums.size();
		vector<int>dp(len, 0);
		dp[0] = nums[0];
		int maxnum = dp[0];

		for (int i = 1;i < len;i++) {
			if (dp[i - 1] > 0) {
				dp[i] = dp[i - 1] + nums[i];
			}
			else{
				dp[i] = nums[i];
			}
			maxnum = max(maxnum, dp[i]);
		}
		return maxnum;
	}
};

int main5() {
	vector<int>arr{-2,1,-3,4,-1,2,1,-5,-4};
	for (int i = 0;i < arr.size();i++) {
		cout << arr[i] << ',';
	}
	cout << endl;

	Solution F;
	int maxnum = F.maxSubArray(arr);
	cout << "最大子列和为：" << maxnum << endl;
	system("pause");
	return 0;
}