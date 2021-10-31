package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
)

// dchkeq tests Dgeequ, Dgbequ, Dpoequ, Dppequ and Dpbequ
func dchkeq(thresh float64, t *testing.T) {
	var ok bool
	var ccond, eps, norm, one, ratio, rcmax, rcmin, rcond, ten, zero float64
	var i, info, j, kl, ku, m, n, npow, nsz, nszb, nszp int
	var err error

	zero = 0.0
	one = 1.0
	ten = 1.0e1
	nsz = 5
	nszb = 3*nsz - 2
	nszp = (nsz * (nsz + 1)) / 2
	npow = 2*nsz + 1
	ap := vf(nszp)
	c := vf(nsz)
	pow := vf(npow)
	r := vf(nsz)
	reslts := make([]float64, 5)
	rpow := vf(npow)
	a := mf(nsz, nsz, opts)
	ab := mf(nszb, nszb, opts)

	path := "Deq"
	alasumStart(path)

	eps = golapack.Dlamch(Precision)
	for i = 1; i <= 5; i++ {
		reslts[i-1] = zero
	}
	for i = 1; i <= npow; i++ {
		pow.Set(i-1, math.Pow(ten, float64(i-1)))
		rpow.Set(i-1, one/pow.Get(i-1))
	}

	//     Test Dgeequ
	for n = 0; n <= nsz; n++ {
		for m = 0; m <= nsz; m++ {

			for j = 1; j <= nsz; j++ {
				for i = 1; i <= nsz; i++ {
					if i <= m && j <= n {
						a.Set(i-1, j-1, pow.Get(i+j)*math.Pow(-1, float64(i+j)))
					} else {
						a.Set(i-1, j-1, zero)
					}
				}
			}

			if rcond, ccond, norm, info, err = golapack.Dgeequ(m, n, a, r, c); err != nil {
				panic(err)
			}

			if info != 0 {
				reslts[0] = one
			} else {
				if n != 0 && m != 0 {
					reslts[0] = math.Max(reslts[0], math.Abs((rcond-rpow.Get(m-1))/rpow.Get(m-1)))
					reslts[0] = math.Max(reslts[0], math.Abs((ccond-rpow.Get(n-1))/rpow.Get(n-1)))
					reslts[0] = math.Max(reslts[0], math.Abs((norm-pow.Get(n+m))/pow.Get(n+m)))
					for i = 1; i <= m; i++ {
						reslts[0] = math.Max(reslts[0], math.Abs((r.Get(i-1)-rpow.Get(i+n))/rpow.Get(i+n)))
					}
					for j = 1; j <= n; j++ {
						reslts[0] = math.Max(reslts[0], math.Abs((c.Get(j-1)-pow.Get(n-j))/pow.Get(n-j)))
					}
				}
			}

		}
	}

	//     Test with zero rows and columns
	for j = 1; j <= nsz; j++ {
		a.Set(max(nsz-1, 1)-1, j-1, zero)
	}
	if rcond, ccond, norm, info, err = golapack.Dgeequ(nsz, nsz, a, r, c); err != nil {
		panic(err)
	}
	if info != max(nsz-1, 1) {
		reslts[0] = one
	}

	for j = 1; j <= nsz; j++ {
		a.Set(max(nsz-1, 1)-1, j-1, one)
	}
	for i = 1; i <= nsz; i++ {
		a.Set(i-1, max(nsz-1, 1)-1, zero)
	}
	if rcond, ccond, norm, info, err = golapack.Dgeequ(nsz, nsz, a, r, c); err != nil {
		panic(err)
	}
	if info != nsz+max(nsz-1, 1) {
		reslts[0] = one
	}
	reslts[0] = reslts[0] / eps

	//     Test Dgbequ
	for n = 0; n <= nsz; n++ {
		for m = 0; m <= nsz; m++ {
			for kl = 0; kl <= max(m-1, 0); kl++ {
				for ku = 0; ku <= max(n-1, 0); ku++ {

					for j = 1; j <= nsz; j++ {
						for i = 1; i <= nszb; i++ {
							ab.Set(i-1, j-1, zero)
						}
					}
					for j = 1; j <= n; j++ {
						for i = 1; i <= m; i++ {
							if i <= min(m, j+kl) && i >= max(1, j-ku) && j <= n {
								ab.Set(ku+1+i-j-1, j-1, pow.Get(i+j)*math.Pow(-1, float64(i+j)))
							}
						}
					}

					if rcond, ccond, norm, info, err = golapack.Dgbequ(m, n, kl, ku, ab, r, c); err != nil {
						panic(err)
					}

					if info != 0 {
						if !((n+kl < m && info == n+kl+1) || (m+ku < n && info == 2*m+ku+1)) {
							reslts[1] = one
						}
					} else {
						if n != 0 && m != 0 {

							rcmin = r.Get(0)
							rcmax = r.Get(0)
							for i = 1; i <= m; i++ {
								rcmin = math.Min(rcmin, r.Get(i-1))
								rcmax = math.Max(rcmax, r.Get(i-1))
							}
							ratio = rcmin / rcmax
							reslts[1] = math.Max(reslts[1], math.Abs((rcond-ratio)/ratio))

							rcmin = c.Get(0)
							rcmax = c.Get(0)
							for j = 1; j <= n; j++ {
								rcmin = math.Min(rcmin, c.Get(j-1))
								rcmax = math.Max(rcmax, c.Get(j-1))
							}
							ratio = rcmin / rcmax
							reslts[1] = math.Max(reslts[1], math.Abs((ccond-ratio)/ratio))

							reslts[1] = math.Max(reslts[1], math.Abs((norm-pow.Get(n+m))/pow.Get(n+m)))
							for i = 1; i <= m; i++ {
								rcmax = zero
								for j = 1; j <= n; j++ {
									if i <= j+kl && i >= j-ku {
										ratio = math.Abs(r.Get(i-1) * pow.Get(i+j) * c.Get(j-1))
										rcmax = math.Max(rcmax, ratio)
									}
								}
								reslts[1] = math.Max(reslts[1], math.Abs(one-rcmax))
							}

							for j = 1; j <= n; j++ {
								rcmax = zero
								for i = 1; i <= m; i++ {
									if i <= j+kl && i >= j-ku {
										ratio = math.Abs(r.Get(i-1) * pow.Get(i+j) * c.Get(j-1))
										rcmax = math.Max(rcmax, ratio)
									}
								}
								reslts[1] = math.Max(reslts[1], math.Abs(one-rcmax))
							}
						}
					}

				}
			}
		}
	}
	reslts[1] = reslts[1] / eps

	//     Test Dpoequ
	for n = 0; n <= nsz; n++ {

		for i = 1; i <= nsz; i++ {
			for j = 1; j <= nsz; j++ {
				if i <= n && j == i {
					a.Set(i-1, j-1, pow.Get(i+j)*math.Pow(-1, float64(i+j)))
				} else {
					a.Set(i-1, j-1, zero)
				}
			}
		}

		if rcond, norm, info, err = golapack.Dpoequ(n, a, r); err != nil {
			panic(err)
		}

		if info != 0 {
			reslts[2] = one
		} else {
			if n != 0 {
				reslts[2] = math.Max(reslts[2], math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1)))
				reslts[2] = math.Max(reslts[2], math.Abs((norm-pow.Get(2*n))/pow.Get(2*n)))
				for i = 1; i <= n; i++ {
					reslts[2] = math.Max(reslts[2], math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i)))
				}
			}
		}
	}
	a.Set(max(nsz-1, 1)-1, max(nsz-1, 1)-1, -one)
	if rcond, norm, info, err = golapack.Dpoequ(nsz, a, r); err != nil {
		panic(err)
	}
	if info != max(nsz-1, 1) {
		reslts[2] = one
	}
	reslts[2] = reslts[2] / eps

	//     Test Dppequ
	for n = 0; n <= nsz; n++ {
		//        Upper triangular packed storage
		for i = 1; i <= (n*(n+1))/2; i++ {
			ap.Set(i-1, zero)
		}
		for i = 1; i <= n; i++ {
			ap.Set((i*(i+1))/2-1, pow.Get(2*i))
		}

		if rcond, norm, info, err = golapack.Dppequ(Upper, n, ap, r); err != nil {
			panic(err)
		}

		if info != 0 {
			reslts[3] = one
		} else {
			if n != 0 {
				reslts[3] = math.Max(reslts[3], math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1)))
				reslts[3] = math.Max(reslts[3], math.Abs((norm-pow.Get(2*n))/pow.Get(2*n)))
				for i = 1; i <= n; i++ {
					reslts[3] = math.Max(reslts[3], math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i)))
				}
			}
		}

		//        Lower triangular packed storage
		for i = 1; i <= (n*(n+1))/2; i++ {
			ap.Set(i-1, zero)
		}
		j = 1
		for i = 1; i <= n; i++ {
			ap.Set(j-1, pow.Get(2*i))
			j = j + (n - i + 1)
		}

		if rcond, norm, info, err = golapack.Dppequ(Lower, n, ap, r); err != nil {
			panic(err)
		}

		if info != 0 {
			reslts[3] = one
		} else {
			if n != 0 {
				reslts[3] = math.Max(reslts[3], math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1)))
				reslts[3] = math.Max(reslts[3], math.Abs((norm-pow.Get(2*n))/pow.Get(2*n)))
				for i = 1; i <= n; i++ {
					reslts[3] = math.Max(reslts[3], math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i)))
				}
			}
		}

	}
	i = (nsz*(nsz+1))/2 - 2
	ap.Set(i-1, -one)
	if rcond, norm, info, err = golapack.Dppequ(Lower, nsz, ap, r); err != nil {
		panic(err)
	}
	if info != max(nsz-1, 1) {
		reslts[3] = one
	}
	reslts[3] = reslts[3] / eps

	//     Test Dpbequ
	for n = 0; n <= nsz; n++ {
		for kl = 0; kl <= max(n-1, 0); kl++ {
			//           Test upper triangular storage
			for j = 1; j <= nsz; j++ {
				for i = 1; i <= nszb; i++ {
					ab.Set(i-1, j-1, zero)
				}
			}
			for j = 1; j <= n; j++ {
				ab.Set(kl, j-1, pow.Get(2*j))
			}

			if rcond, norm, info, err = golapack.Dpbequ(Upper, n, kl, ab, r); err != nil {
				panic(err)
			}

			if info != 0 {
				reslts[4] = one
			} else {
				if n != 0 {
					reslts[4] = math.Max(reslts[4], math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1)))
					reslts[4] = math.Max(reslts[4], math.Abs((norm-pow.Get(2*n))/pow.Get(2*n)))
					for i = 1; i <= n; i++ {
						reslts[4] = math.Max(reslts[4], math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i)))
					}
				}
			}
			if n != 0 {
				ab.Set(kl, max(n-1, 1)-1, -one)
				if rcond, norm, info, err = golapack.Dpbequ(Upper, n, kl, ab, r); err != nil {
					panic(err)
				}
				if info != max(n-1, 1) {
					reslts[4] = one
				}
			}

			//           Test lower triangular storage
			for j = 1; j <= nsz; j++ {
				for i = 1; i <= nszb; i++ {
					ab.Set(i-1, j-1, zero)
				}
			}
			for j = 1; j <= n; j++ {
				ab.Set(0, j-1, pow.Get(2*j))
			}

			if rcond, norm, info, err = golapack.Dpbequ(Lower, n, kl, ab, r); err != nil {
				panic(err)
			}

			if info != 0 {
				reslts[4] = one
			} else {
				if n != 0 {
					reslts[4] = math.Max(reslts[4], math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1)))
					reslts[4] = math.Max(reslts[4], math.Abs((norm-pow.Get(2*n))/pow.Get(2*n)))
					for i = 1; i <= n; i++ {
						reslts[4] = math.Max(reslts[4], math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i)))
					}
				}
			}
			if n != 0 {
				ab.Set(0, max(n-1, 1)-1, -one)
				if rcond, norm, info, err = golapack.Dpbequ(Lower, n, kl, ab, r); err != nil {
					panic(err)
				}
				if info != max(n-1, 1) {
					reslts[4] = one
				}
			}
		}
	}
	reslts[4] = reslts[4] / eps
	ok = (reslts[0] <= thresh) && (reslts[1] <= thresh) && (reslts[2] <= thresh) && (reslts[3] <= thresh) && (reslts[4] <= thresh)
	if ok {
		// fmt.Printf(" All tests for %3s routines passed the threshold\n\n", path)
		fmt.Printf("Pass\n")
	} else {
		if reslts[0] > thresh {
			t.Fail()
			fmt.Printf(" Dgeequ failed test with value %10.3E exceeding threshold %10.3E\n\n", reslts[0], thresh)
		}
		if reslts[1] > thresh {
			t.Fail()
			fmt.Printf(" Dgbequ failed test with value %10.3E exceeding threshold %10.3E\n\n", reslts[1], thresh)
		}
		if reslts[2] > thresh {
			t.Fail()
			fmt.Printf(" Dpoequ failed test with value %10.3E exceeding threshold %10.3E\n\n", reslts[2], thresh)
		}
		if reslts[3] > thresh {
			t.Fail()
			fmt.Printf(" Dppequ failed test with value %10.3E exceeding threshold %10.3E\n\n", reslts[3], thresh)
		}
		if reslts[4] > thresh {
			t.Fail()
			fmt.Printf(" Dpbequ failed test with value %10.3E exceeding threshold %10.3E\n\n", reslts[4], thresh)
		}
	}
}
