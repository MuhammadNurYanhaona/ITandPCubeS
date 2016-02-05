#ifndef COMMON_H_
#define COMMON_H_

int gcd(int a, int b) {
	while (a != b) {
		if (a > b) a = a - b;
		else b = b - a;
	}
	return a;
}

int lcm(int a, int b) {
	return (a * b) / gcd(a, b);
}


#endif /* COMMON_H_ */
