package lin

import (
	"fmt"
	"golinalg/golapack"
	"math"
	"testing"
)

// Zchkeq tests ZGEEQU, ZGBEQU, ZPOEQU, ZPPEQU and ZPBEQU
func Zchkeq(thresh *float64, nout *int, t *testing.T) {
	var ok bool
	var cone, czero complex128
	var ccond, eps, norm, one, ratio, rcmax, rcmin, rcond, ten, zero float64
	var i, info, j, kl, ku, m, n, npow, nsz, nszb, nszp int

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

	path := []byte("ZEQ")

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
						a.SetRe(i-1, j-1, pow.Get(i+j+1-1)*math.Pow(float64(-1), float64(i+j)))
					} else {
						a.Set(i-1, j-1, czero)
					}
				}
			}

			golapack.Zgeequ(&m, &n, a, &nsz, r, c, &rcond, &ccond, &norm, &info)

			if info != 0 {
				reslts.Set(0, one)
			} else {
				if n != 0 && m != 0 {
					reslts.Set(0, maxf64(reslts.Get(0), math.Abs((rcond-rpow.Get(m-1))/rpow.Get(m-1))))
					reslts.Set(0, maxf64(reslts.Get(0), math.Abs((ccond-rpow.Get(n-1))/rpow.Get(n-1))))
					reslts.Set(0, maxf64(reslts.Get(0), math.Abs((norm-pow.Get(n+m+1-1))/pow.Get(n+m+1-1))))
					for i = 1; i <= m; i++ {
						reslts.Set(0, maxf64(reslts.Get(0), math.Abs((r.Get(i-1)-rpow.Get(i+n+1-1))/rpow.Get(i+n+1-1))))
					}
					for j = 1; j <= n; j++ {
						reslts.Set(0, maxf64(reslts.Get(0), math.Abs((c.Get(j-1)-pow.Get(n-j+1-1))/pow.Get(n-j+1-1))))
					}
				}
			}

		}
	}

	//     Test with zero rows and columns
	for j = 1; j <= nsz; j++ {
		a.Set(maxint(nsz-1, 1)-1, j-1, czero)
	}
	golapack.Zgeequ(&nsz, &nsz, a, &nsz, r, c, &rcond, &ccond, &norm, &info)
	if info != maxint(nsz-1, 1) {
		reslts.Set(0, one)
	}

	for j = 1; j <= nsz; j++ {
		a.Set(maxint(nsz-1, 1)-1, j-1, cone)
	}
	for i = 1; i <= nsz; i++ {
		a.Set(i-1, maxint(nsz-1, 1)-1, czero)
	}
	golapack.Zgeequ(&nsz, &nsz, a, &nsz, r, c, &rcond, &ccond, &norm, &info)
	if info != nsz+maxint(nsz-1, 1) {
		reslts.Set(0, one)
	}
	reslts.Set(0, reslts.Get(0)/eps)

	//     Test ZGBEQU
	for n = 0; n <= nsz; n++ {
		for m = 0; m <= nsz; m++ {
			for kl = 0; kl <= maxint(m-1, 0); kl++ {
				for ku = 0; ku <= maxint(n-1, 0); ku++ {
					//
					for j = 1; j <= nsz; j++ {
						for i = 1; i <= nszb; i++ {
							ab.Set(i-1, j-1, czero)
						}
					}
					for j = 1; j <= n; j++ {
						for i = 1; i <= m; i++ {
							if i <= minint(m, j+kl) && i >= maxint(1, j-ku) && j <= n {
								ab.SetRe(ku+1+i-j-1, j-1, pow.Get(i+j+1-1)*math.Pow(float64(-1), float64(i+j)))
							}
						}
					}

					golapack.Zgbequ(&m, &n, &kl, &ku, ab, &nszb, r, c, &rcond, &ccond, &norm, &info)

					if info != 0 {
						if !((n+kl < m && info == n+kl+1) || (m+ku < n && info == 2*m+ku+1)) {
							reslts.Set(1, one)
						}
					} else {
						if n != 0 && m != 0 {

							rcmin = r.Get(0)
							rcmax = r.Get(0)
							for i = 1; i <= m; i++ {
								rcmin = minf64(rcmin, r.Get(i-1))
								rcmax = maxf64(rcmax, r.Get(i-1))
							}
							ratio = rcmin / rcmax
							reslts.Set(1, maxf64(reslts.Get(1), math.Abs((rcond-ratio)/ratio)))

							rcmin = c.Get(0)
							rcmax = c.Get(0)
							for j = 1; j <= n; j++ {
								rcmin = minf64(rcmin, c.Get(j-1))
								rcmax = maxf64(rcmax, c.Get(j-1))
							}
							ratio = rcmin / rcmax
							reslts.Set(1, maxf64(reslts.Get(1), math.Abs((ccond-ratio)/ratio)))

							reslts.Set(1, maxf64(reslts.Get(1), math.Abs((norm-pow.Get(n+m+1-1))/pow.Get(n+m+1-1))))
							for i = 1; i <= m; i++ {
								rcmax = zero
								for j = 1; j <= n; j++ {
									if i <= j+kl && i >= j-ku {
										ratio = math.Abs(r.Get(i-1) * pow.Get(i+j+1-1) * c.Get(j-1))
										rcmax = maxf64(rcmax, ratio)
									}
								}
								reslts.Set(1, maxf64(reslts.Get(1), math.Abs(one-rcmax)))
							}

							for j = 1; j <= n; j++ {
								rcmax = zero
								for i = 1; i <= m; i++ {
									if i <= j+kl && i >= j-ku {
										ratio = math.Abs(r.Get(i-1) * pow.Get(i+j+1-1) * c.Get(j-1))
										rcmax = maxf64(rcmax, ratio)
									}
								}
								reslts.Set(1, maxf64(reslts.Get(1), math.Abs(one-rcmax)))
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
					a.SetRe(i-1, j-1, pow.Get(i+j+1-1)*math.Pow(float64(-1), float64(i+j)))
				} else {
					a.Set(i-1, j-1, czero)
				}
			}
		}

		golapack.Zpoequ(&n, a, &nsz, r, &rcond, &norm, &info)

		if info != 0 {
			reslts.Set(2, one)
		} else {
			if n != 0 {
				reslts.Set(2, maxf64(reslts.Get(2), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
				reslts.Set(2, maxf64(reslts.Get(2), math.Abs((norm-pow.Get(2*n+1-1))/pow.Get(2*n+1-1))))
				for i = 1; i <= n; i++ {
					reslts.Set(2, maxf64(reslts.Get(2), math.Abs((r.Get(i-1)-rpow.Get(i+1-1))/rpow.Get(i+1-1))))
				}
			}
		}
	}
	a.Set(maxint(nsz-1, 1)-1, maxint(nsz-1, 1)-1, -cone)
	golapack.Zpoequ(&nsz, a, &nsz, r, &rcond, &norm, &info)
	if info != maxint(nsz-1, 1) {
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
			ap.SetRe((i*(i+1))/2-1, pow.Get(2*i+1-1))
		}

		golapack.Zppequ('U', &n, ap, r, &rcond, &norm, &info)

		if info != 0 {
			reslts.Set(3, one)
		} else {
			if n != 0 {
				reslts.Set(3, maxf64(reslts.Get(3), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
				reslts.Set(3, maxf64(reslts.Get(3), math.Abs((norm-pow.Get(2*n+1-1))/pow.Get(2*n+1-1))))
				for i = 1; i <= n; i++ {
					reslts.Set(3, maxf64(reslts.Get(3), math.Abs((r.Get(i-1)-rpow.Get(i+1-1))/rpow.Get(i+1-1))))
				}
			}
		}

		//        Lower triangular packed storage
		for i = 1; i <= (n*(n+1))/2; i++ {
			ap.Set(i-1, czero)
		}
		j = 1
		for i = 1; i <= n; i++ {
			ap.SetRe(j-1, pow.Get(2*i+1-1))
			j = j + (n - i + 1)
		}

		golapack.Zppequ('L', &n, ap, r, &rcond, &norm, &info)

		if info != 0 {
			reslts.Set(3, one)
		} else {
			if n != 0 {
				reslts.Set(3, maxf64(reslts.Get(3), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
				reslts.Set(3, maxf64(reslts.Get(3), math.Abs((norm-pow.Get(2*n+1-1))/pow.Get(2*n+1-1))))
				for i = 1; i <= n; i++ {
					reslts.Set(3, maxf64(reslts.Get(3), math.Abs((r.Get(i-1)-rpow.Get(i+1-1))/rpow.Get(i+1-1))))
				}
			}
		}

	}
	i = (nsz*(nsz+1))/2 - 2
	ap.Set(i-1, -cone)
	golapack.Zppequ('L', &nsz, ap, r, &rcond, &norm, &info)
	if info != maxint(nsz-1, 1) {
		reslts.Set(3, one)
	}
	reslts.Set(3, reslts.Get(3)/eps)

	//     Test ZPBEQU
	for n = 0; n <= nsz; n++ {
		for kl = 0; kl <= maxint(n-1, 0); kl++ {
			//           Test upper triangular storage
			for j = 1; j <= nsz; j++ {
				for i = 1; i <= nszb; i++ {
					ab.Set(i-1, j-1, czero)
				}
			}
			for j = 1; j <= n; j++ {
				ab.SetRe(kl+1-1, j-1, pow.Get(2*j+1-1))
			}

			golapack.Zpbequ('U', &n, &kl, ab, &nszb, r, &rcond, &norm, &info)

			if info != 0 {
				reslts.Set(4, one)
			} else {
				if n != 0 {
					reslts.Set(4, maxf64(reslts.Get(4), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
					reslts.Set(4, maxf64(reslts.Get(4), math.Abs((norm-pow.Get(2*n+1-1))/pow.Get(2*n+1-1))))
					for i = 1; i <= n; i++ {
						reslts.Set(4, maxf64(reslts.Get(4), math.Abs((r.Get(i-1)-rpow.Get(i+1-1))/rpow.Get(i+1-1))))
					}
				}
			}
			if n != 0 {
				ab.Set(kl+1-1, maxint(n-1, 1)-1, -cone)
				golapack.Zpbequ('U', &n, &kl, ab, &nszb, r, &rcond, &norm, &info)
				if info != maxint(n-1, 1) {
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
				ab.SetRe(0, j-1, pow.Get(2*j+1-1))
			}

			golapack.Zpbequ('L', &n, &kl, ab, &nszb, r, &rcond, &norm, &info)

			if info != 0 {
				reslts.Set(4, one)
			} else {
				if n != 0 {
					reslts.Set(4, maxf64(reslts.Get(4), math.Abs((rcond-rpow.Get(n-1))/rpow.Get(n-1))))
					reslts.Set(4, maxf64(reslts.Get(4), math.Abs((norm-pow.Get(2*n+1-1))/pow.Get(2*n+1-1))))
					for i = 1; i <= n; i++ {
						reslts.Set(4, maxf64(reslts.Get(4), math.Abs((r.Get(i-1)-rpow.Get(i+1-1))/rpow.Get(i+1-1))))
					}
				}
			}
			if n != 0 {
				ab.Set(0, maxint(n-1, 1)-1, -cone)
				golapack.Zpbequ('L', &n, &kl, ab, &nszb, r, &rcond, &norm, &info)
				if info != maxint(n-1, 1) {
					reslts.Set(4, one)
				}
			}
		}
	}
	reslts.Set(4, reslts.Get(4)/eps)
	ok = (reslts.Get(0) <= (*thresh)) && (reslts.Get(1) <= (*thresh)) && (reslts.Get(2) <= (*thresh)) && (reslts.Get(3) <= (*thresh)) && (reslts.Get(4) <= (*thresh))
	if ok {
		fmt.Printf(" All tests for %3s routines passed the threshold\n\n", path)
	} else {
		if reslts.Get(0) > (*thresh) {
			fmt.Printf(" ZGEEQU failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(0), *thresh)
		}
		if reslts.Get(1) > (*thresh) {
			fmt.Printf(" ZGBEQU failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(1), *thresh)
		}
		if reslts.Get(2) > (*thresh) {
			fmt.Printf(" ZPOEQU failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(2), *thresh)
		}
		if reslts.Get(3) > (*thresh) {
			fmt.Printf(" ZPPEQU failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(3), *thresh)
		}
		if reslts.Get(4) > (*thresh) {
			fmt.Printf(" ZPBEQU failed test with value %10.3E exceeding threshold %10.3E\n", reslts.Get(4), *thresh)
		}
	}
}
