package lin

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
)

// zchkeq tests ZGEEQU, ZGBEQU, ZPOEQU, ZPPEQU and ZPBEQU
func zchkeq(thresh float64, t *testing.T) {
	var ok bool
	var cone, czero complex128
	var ccond, eps, norm, one, ratio, rcmax, rcmin, rcond, ten, zero float64
	var i, info, j, kl, ku, m, n, npow, nsz, nszb, nszp int
	var err error

	zero = 0.0
	one = 1.0
	ten = 1.0e1
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	nsz = 5
	nszb = 3*nsz - 2
	nszp = (nsz * (nsz + 1)) / 2
	npow = 2*nsz + 1
	ap := cvf(nszp)
	c := vf(5)
	pow := vf(npow)
	r := vf(5)
	reslts := vf(5)
	rpow := vf(npow)
	a := cmf(5, 5, opts)
	ab := cmf(nszb, 5, opts)

	path := "Zeq"
	alasumStart(path)

	eps = golapack.Dlamch(Precision)
	for i = 1; i <= 5; i++ {
		reslts.Set(i-1, zero)
	}
	for i = 1; i <= npow; i++ {
		pow.Set(i-1, math.Pow(ten, float64(i-1)))
		rpow.Set(i-1, one/pow.Get(i-1))
	}

	//     Test ZGEEQU
	for n = 0; n <= nsz; n++ {
		for m = 0; m <= nsz; m++ {

			for j = 1; j <= nsz; j++ {
				for i = 1; i <= nsz; i++ {
					if i <= m && j <= n {
						a.SetRe(i-1, j-1, pow.Get(i+j)*math.Pow(float64(-1), float64(i+j)))
					} else {
						a.Set(i-1, j-1, czero)
					}
				}
			}

			if rcond, ccond, norm, info, err = golapack.Zgeequ(m, n, a.Off(0, 0).UpdateRows(nsz), r, c); err != nil || info != 0 {
				reslts.Set(0, one)
			} else {
				if n != 0 && m != 0 {
					reslts.Set(0, math.Max(reslts.Get(0), math.Abs((rcond-rpow.Get(m-1))/rpow.Get(m-1))))
					reslts.Set(0, math.Max(reslts.Get(0), math.Abs((ccond-rpow.Get(n-1))/rpow.Get(n-1))))
					reslts.Set(0, math.Max(reslts.Get(0), math.Abs((norm-pow.Get(n+m))/pow.Get(n+m))))
					for i = 1; i <= m; i++ {
						reslts.Set(0, math.Max(reslts.Get(0), math.Abs((r.Get(i-1)-rpow.Get(i+n))/rpow.Get(i+n))))
					}
					for j = 1; j <= n; j++ {
						reslts.Set(0, math.Max(reslts.Get(0), math.Abs((c.Get(j-1)-pow.Get(n-j))/pow.Get(n-j))))
					}
				}
			}

		}
	}

	//     Test with zero rows and columns
	for j = 1; j <= nsz; j++ {
		a.Set(max(nsz-1, 1)-1, j-1, czero)
	}
	if rcond, ccond, norm, info, err = golapack.Zgeequ(nsz, nsz, a.Off(0, 0).UpdateRows(nsz), r, c); err != nil || info != max(nsz-1, 1) {
		reslts.Set(0, one)
	}

	for j = 1; j <= nsz; j++ {
		a.Set(max(nsz-1, 1)-1, j-1, cone)
	}
	for i = 1; i <= nsz; i++ {
		a.Set(i-1, max(nsz-1, 1)-1, czero)
	}
	if rcond, ccond, norm, info, err = golapack.Zgeequ(nsz, nsz, a.Off(0, 0).UpdateRows(nsz), r, c); err != nil || info != nsz+max(nsz-1, 1) {
		reslts.Set(0, one)
	}
	reslts.Set(0, reslts.Get(0)/eps)

	//     Test ZGBEQU
	for n = 0; n <= nsz; n++ {
		for m = 0; m <= nsz; m++ {
			for kl = 0; kl <= max(m-1, 0); kl++ {
				for ku = 0; ku <= max(n-1, 0); ku++ {
					//
					for j = 1; j <= nsz; j++ {
						for i = 1; i <= nszb; i++ {
							ab.Set(i-1, j-1, czero)
						}
					}
					for j = 1; j <= n; j++ {
						for i = 1; i <= m; i++ {
							if i <= min(m, j+kl) && i >= max(1, j-ku) && j <= n {
								ab.SetRe(ku+1+i-j-1, j-1, pow.Get(i+j)*math.Pow(float64(-1), float64(i+j)))
							}
						}
					}

					if rcond, ccond, norm, info, err = golapack.Zgbequ(m, n, kl, ku, ab, r, c); err != nil {
						panic(err)
					}

					if info != 0 {
						if !((n+kl < m && info == n+kl+1) || (m+ku < n && info == 2*m+ku+1)) {
							reslts.Set(1, one)
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
							reslts.Set(1, math.Max(reslts.Get(1), math.Abs((rcond-ratio)/ratio)))

							rcmin = c.Get(0)
							rcmax = c.Get(0)
							for j = 1; j <= n; j++ {
								rcmin = math.Min(rcmin, c.Get(j-1))
								rcmax = math.Max(rcmax, c.Get(j-1))
							}
							ratio = rcmin / rcmax
							reslts.Set(1, math.Max(reslts.Get(1), math.Abs((ccond-ratio)/ratio)))

							reslts.Set(1, math.Max(reslts.Get(1), math.Abs((norm-pow.Get(n+m))/pow.Get(n+m))))
							for i = 1; i <= m; i++ {
								rcmax = zero
								for j = 1; j <= n; j++ {
									if i <= j+kl && i >= j-ku {
										ratio = math.Abs(r.Get(i-1) * pow.Get(i+j) * c.Get(j-1))
										rcmax = math.Max(rcmax, ratio)
									}
								}
								reslts.Set(1, math.Max(reslts.Get(1), math.Abs(one-rcmax)))
							}

							for j = 1; j <= n; j++ {
								rcmax = zero
								for i = 1; i <= m; i++ {
									if i <= j+kl && i >= j-ku {
										ratio = math.Abs(r.Get(i-1) * pow.Get(i+j) * c.Get(j-1))
										rcmax = math.Max(rcmax, ratio)
									}
								}
								reslts.Set(1, math.Max(reslts.Get(1), math.Abs(one-rcmax)))
							}
						}
					}

				}
			}
		}
	}
	reslts.Set(1, reslts.Get(1)/eps)

	//     Test ZPOEQU
	for n = 0; n <= nsz; n++ {

		for i = 1; i <= nsz; i++ {
			for j = 1; j <= nsz; j++ {
				if i <= n && j == i {
					a.SetRe(i-1, j-1, pow.Get(i+j)*math.Pow(float64(-1), float64(i+j)))
				} else {
					a.Set(i-1, j-1, czero)
				}
			}
		}

		if rcond, norm, info, err = golapack.Zpoequ(n, a, r); err != nil {
			panic(err)
		}

		if info != 0 {
			reslts.Set(2, one)
		} else {
			if n != 0 {
				reslts.Set(2, math.Max(reslts.Get(2), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
				reslts.Set(2, math.Max(reslts.Get(2), math.Abs((norm-pow.Get(2*n))/pow.Get(2*n))))
				for i = 1; i <= n; i++ {
					reslts.Set(2, math.Max(reslts.Get(2), math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i))))
				}
			}
		}
	}
	a.Set(max(nsz-1, 1)-1, max(nsz-1, 1)-1, -cone)
	if rcond, norm, info, err = golapack.Zpoequ(nsz, a, r); err != nil {
		panic(err)
	}
	if info != max(nsz-1, 1) {
		reslts.Set(2, one)
	}
	reslts.Set(2, reslts.Get(2)/eps)

	//     Test ZPPEQU
	for n = 0; n <= nsz; n++ {
		//        Upper triangular packed storage
		for i = 1; i <= (n*(n+1))/2; i++ {
			ap.Set(i-1, czero)
		}
		for i = 1; i <= n; i++ {
			ap.SetRe((i*(i+1))/2-1, pow.Get(2*i))
		}

		if rcond, norm, info, err = golapack.Zppequ(Upper, n, ap, r); err != nil || info != 0 {
			reslts.Set(3, one)
		} else {
			if n != 0 {
				reslts.Set(3, math.Max(reslts.Get(3), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
				reslts.Set(3, math.Max(reslts.Get(3), math.Abs((norm-pow.Get(2*n))/pow.Get(2*n))))
				for i = 1; i <= n; i++ {
					reslts.Set(3, math.Max(reslts.Get(3), math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i))))
				}
			}
		}

		//        Lower triangular packed storage
		for i = 1; i <= (n*(n+1))/2; i++ {
			ap.Set(i-1, czero)
		}
		j = 1
		for i = 1; i <= n; i++ {
			ap.SetRe(j-1, pow.Get(2*i))
			j = j + (n - i + 1)
		}

		if rcond, norm, info, err = golapack.Zppequ(Lower, n, ap, r); err != nil || info != 0 {
			reslts.Set(3, one)
		} else {
			if n != 0 {
				reslts.Set(3, math.Max(reslts.Get(3), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
				reslts.Set(3, math.Max(reslts.Get(3), math.Abs((norm-pow.Get(2*n))/pow.Get(2*n))))
				for i = 1; i <= n; i++ {
					reslts.Set(3, math.Max(reslts.Get(3), math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i))))
				}
			}
		}

	}
	i = (nsz*(nsz+1))/2 - 2
	ap.Set(i-1, -cone)
	if rcond, norm, info, err = golapack.Zppequ(Lower, nsz, ap, r); err != nil || info != max(nsz-1, 1) {
		reslts.Set(3, one)
	}
	reslts.Set(3, reslts.Get(3)/eps)

	//     Test ZPBEQU
	for n = 0; n <= nsz; n++ {
		for kl = 0; kl <= max(n-1, 0); kl++ {
			//           Test upper triangular storage
			for j = 1; j <= nsz; j++ {
				for i = 1; i <= nszb; i++ {
					ab.Set(i-1, j-1, czero)
				}
			}
			for j = 1; j <= n; j++ {
				ab.SetRe(kl, j-1, pow.Get(2*j))
			}

			if rcond, norm, info, err = golapack.Zpbequ(Upper, n, kl, ab, r); err != nil {
				panic(err)
			}

			if info != 0 {
				reslts.Set(4, one)
			} else {
				if n != 0 {
					reslts.Set(4, math.Max(reslts.Get(4), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
					reslts.Set(4, math.Max(reslts.Get(4), math.Abs((norm-pow.Get(2*n))/pow.Get(2*n))))
					for i = 1; i <= n; i++ {
						reslts.Set(4, math.Max(reslts.Get(4), math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i))))
					}
				}
			}
			if n != 0 {
				ab.Set(kl, max(n-1, 1)-1, -cone)
				if rcond, norm, info, err = golapack.Zpbequ(Upper, n, kl, ab, r); err != nil {
					panic(err)
				}
				if info != max(n-1, 1) {
					reslts.Set(4, one)
				}
			}

			//           Test lower triangular storage
			for j = 1; j <= nsz; j++ {
				for i = 1; i <= nszb; i++ {
					ab.Set(i-1, j-1, czero)
				}
			}
			for j = 1; j <= n; j++ {
				ab.SetRe(0, j-1, pow.Get(2*j))
			}

			if rcond, norm, info, err = golapack.Zpbequ(Lower, n, kl, ab, r); err != nil {
				panic(err)
			}

			if info != 0 {
				reslts.Set(4, one)
			} else {
				if n != 0 {
					reslts.Set(4, math.Max(reslts.Get(4), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
					reslts.Set(4, math.Max(reslts.Get(4), math.Abs((norm-pow.Get(2*n))/pow.Get(2*n))))
					for i = 1; i <= n; i++ {
						reslts.Set(4, math.Max(reslts.Get(4), math.Abs((r.Get(i-1)-rpow.Get(i))/rpow.Get(i))))
					}
				}
			}
			if n != 0 {
				ab.Set(0, max(n-1, 1)-1, -cone)
				if rcond, norm, info, err = golapack.Zpbequ(Lower, n, kl, ab, r); err != nil {
					panic(err)
				}
				if info != max(n-1, 1) {
					reslts.Set(4, one)
				}
			}
		}
	}
	reslts.Set(4, reslts.Get(4)/eps)
	ok = (reslts.Get(0) <= thresh) && (reslts.Get(1) <= thresh) && (reslts.Get(2) <= thresh) && (reslts.Get(3) <= thresh) && (reslts.Get(4) <= thresh)
	if ok {
		// fmt.Printf(" All tests for %3s routines passed the threshold\n\n", path)
		fmt.Printf("Pass\n")
	} else {
		if reslts.Get(0) > thresh {
			fmt.Printf(" Zgeequ failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(0), thresh)
		}
		if reslts.Get(1) > thresh {
			fmt.Printf(" Zgbequ failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(1), thresh)
		}
		if reslts.Get(2) > thresh {
			fmt.Printf(" Zpoequ failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(2), thresh)
		}
		if reslts.Get(3) > thresh {
			fmt.Printf(" Zppequ failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(3), thresh)
		}
		if reslts.Get(4) > thresh {
			fmt.Printf(" Zpbequ failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(4), thresh)
		}
	}
}
