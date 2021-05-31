package eig

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zdrgvx checks the nonsymmetric generalized eigenvalue problem
// expert driver ZGGEVX.
//
// ZGGEVX computes the generalized eigenvalues, (optionally) the left
// and/or right eigenvectors, (optionally) computes a balancing
// transformation to improve the conditioning, and (optionally)
// reciprocal condition numbers for the eigenvalues and eigenvectors.
//
// When ZDRGVX is called with NSIZE > 0, two types of test matrix pairs
// are generated by the subroutine DLATM6 and test the driver ZGGEVX.
// The test matrices have the known exact condition numbers for
// eigenvalues. For the condition numbers of the eigenvectors
// corresponding the first and last eigenvalues are also know
// ``exactly'' (see ZLATM6).
// For each matrix pair, the following tests will be performed and
// compared with the threshold THRESH.
//
// (1) maxint over all left eigenvalue/-vector pairs (beta/alpha,l) of
//
//    | l**H * (beta A - alpha B) | / ( ulp maxint( |beta A|, |alpha B| ) )
//
//     where l**H is the conjugate tranpose of l.
//
// (2) maxint over all right eigenvalue/-vector pairs (beta/alpha,r) of
//
//       | (beta A - alpha B) r | / ( ulp maxint( |beta A|, |alpha B| ) )
//
// (3) The condition number S(i) of eigenvalues computed by ZGGEVX
//     differs less than a factor THRESH from the exact S(i) (see
//     ZLATM6).
//
// (4) DIF(i) computed by ZTGSNA differs less than a factor 10*THRESH
//     from the exact value (for the 1st and 5th vectors only).
//
// Test Matrices
// =============
//
// Two kinds of test matrix pairs
//          (A, B) = inverse(YH) * (Da, Db) * inverse(X)
// are used in the tests:
//
// 1: Da = 1+a   0    0    0    0    Db = 1   0   0   0   0
//          0   2+a   0    0    0         0   1   0   0   0
//          0    0   3+a   0    0         0   0   1   0   0
//          0    0    0   4+a   0         0   0   0   1   0
//          0    0    0    0   5+a ,      0   0   0   0   1 , and
//
// 2: Da =  1   -1    0    0    0    Db = 1   0   0   0   0
//          1    1    0    0    0         0   1   0   0   0
//          0    0    1    0    0         0   0   1   0   0
//          0    0    0   1+a  1+b        0   0   0   1   0
//          0    0    0  -1-b  1+a ,      0   0   0   0   1 .
//
// In both cases the same inverse(YH) and inverse(X) are used to compute
// (A, B), giving the exact eigenvectors to (A,B) as (YH, X):
//
// YH:  =  1    0   -y    y   -y    X =  1   0  -x  -x   x
//         0    1   -y    y   -y         0   1   x  -x  -x
//         0    0    1    0    0         0   0   1   0   0
//         0    0    0    1    0         0   0   0   1   0
//         0    0    0    0    1,        0   0   0   0   1 , where
//
// a, b, x and y will have all values independently of each other from
// { sqrt(sqrt(ULP)),  0.1,  1,  10,  1/sqrt(sqrt(ULP)) }.
func Zdrgvx(nsize *int, thresh *float64, nout *int, a *mat.CMatrix, lda *int, b, ai, bi *mat.CMatrix, alpha, beta *mat.CVector, vl, vr *mat.CMatrix, ilo, ihi *int, lscale, rscale, s, dtru, dif, diftru *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, liwork *int, result *mat.Vector, bwork *[]bool, info *int, t *testing.T) {
	var abnorm, anorm, bnorm, half, one, ratio1, ratio2, ten, thrsh2, tnth, ulp, ulpinv, zero float64
	var _i, i, iptype, iwa, iwb, iwx, iwy, j, linfo, maxwrk, minwrk, n, nerrs, nmax, nptknt, ntestt int
	weight := cvf(5)

	zero = 0.0
	one = 1.0
	ten = 1.0e+1
	tnth = 1.0e-1
	half = 0.5

	//     Check for errors
	(*info) = 0

	nmax = 5

	if (*nsize) < 0 {
		(*info) = -1
	} else if (*thresh) < zero {
		(*info) = -2
	} else if (*nout) <= 0 {
		(*info) = -4
	} else if (*lda) < 1 || (*lda) < nmax {
		(*info) = -6
	} else if (*liwork) < nmax+2 {
		(*info) = -26
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	minwrk = 1
	if (*info) == 0 && (*lwork) >= 1 {
		minwrk = 2 * nmax * (nmax + 1)
		maxwrk = nmax * (1 + Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, &nmax, func() *int { y := 1; return &y }(), &nmax, func() *int { y := 0; return &y }()))
		maxwrk = maxint(maxwrk, 2*nmax*(nmax+1))
		work.SetRe(0, float64(maxwrk))
	}

	if (*lwork) < minwrk {
		(*info) = -23
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZDRGVX"), -(*info))
		return
	}

	n = 5
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	thrsh2 = ten * (*thresh)
	nerrs = 0
	nptknt = 0
	ntestt = 0

	// if (*nsize) == 0 {
	// 	goto label90
	// }

	//     Parameters used for generating test matrices.
	weight.SetRe(0, tnth)
	weight.SetRe(1, half)
	weight.SetRe(2, one)
	weight.Set(3, toCmplx(one)/weight.Get(1))
	weight.Set(4, toCmplx(one)/weight.Get(0))

	for iptype = 1; iptype <= 2; iptype++ {
		for iwa = 1; iwa <= 5; iwa++ {
			for iwb = 1; iwb <= 5; iwb++ {
				for iwx = 1; iwx <= 5; iwx++ {
					for iwy = 1; iwy <= 5; iwy++ {
						//                    generated a pair of test matrix
						matgen.Zlatm6(&iptype, func() *int { y := 5; return &y }(), a, lda, b, vr, lda, vl, lda, weight.GetPtr(iwa-1), weight.GetPtr(iwb-1), weight.GetPtr(iwx-1), weight.GetPtr(iwy-1), dtru, diftru)

						//                    Compute eigenvalues/eigenvectors of (A, B).
						//                    Compute eigenvalue/eigenvector condition numbers
						//                    using computed eigenvectors.
						golapack.Zlacpy('F', &n, &n, a, lda, ai, lda)
						golapack.Zlacpy('F', &n, &n, b, lda, bi, lda)

						golapack.Zggevx('N', 'V', 'V', 'B', &n, ai, lda, bi, lda, alpha, beta, vl, lda, vr, lda, ilo, ihi, lscale, rscale, &anorm, &bnorm, s, dif, work, lwork, rwork, iwork, bwork, &linfo)
						if linfo != 0 {
							t.Fail()
							fmt.Printf(" ZDRGVX: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d, IWA=%5d, IWB=%5d, IWX=%5d, IWY=%5d\n", "ZGGEVX", linfo, n, iptype, iwa, iwb, iwx, iwy)
							goto label30
						}

						//                    Compute the norm(A, B)
						golapack.Zlacpy('F', &n, &n, ai, lda, work.CMatrix(n, opts), &n)
						golapack.Zlacpy('F', &n, &n, bi, lda, work.CMatrixOff(n*n+1-1, n, opts), &n)
						abnorm = golapack.Zlange('F', &n, toPtr(2*n), work.CMatrix(n, opts), &n, rwork)

						//                    Tests (1) and (2)
						result.Set(0, zero)
						Zget52(true, &n, a, lda, b, lda, vl, lda, alpha, beta, work, rwork, result.Off(0))
						if result.Get(1) > (*thresh) {
							fmt.Printf(" ZDRGVX: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, IWA=%5d, IWB=%5d, IWX=%5d, IWY=%5d\n", "Left", "ZGGEVX", result.Get(1), n, iptype, iwa, iwb, iwx, iwy)
						}

						result.Set(1, zero)
						Zget52(false, &n, a, lda, b, lda, vr, lda, alpha, beta, work, rwork, result.Off(1))
						if result.Get(2) > (*thresh) {
							fmt.Printf(" ZDRGVX: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, JTYPE=%6d, IWA=%5d, IWB=%5d, IWX=%5d, IWY=%5d\n", "Right", "ZGGEVX", result.Get(2), n, iptype, iwa, iwb, iwx, iwy)
						}

						//                    Test (3)
						result.Set(2, zero)
						for i = 1; i <= n; i++ {
							if s.Get(i-1) == zero {
								if dtru.Get(i-1) > abnorm*ulp {
									result.Set(2, ulpinv)
								}
							} else if dtru.Get(i-1) == zero {
								if s.Get(i-1) > abnorm*ulp {
									result.Set(2, ulpinv)
								}
							} else {
								rwork.Set(i-1, maxf64(math.Abs(dtru.Get(i-1)/s.Get(i-1)), math.Abs(s.Get(i-1)/dtru.Get(i-1))))
								result.Set(2, maxf64(result.Get(2), rwork.Get(i-1)))
							}
						}

						//                    Test (4)
						result.Set(3, zero)
						if dif.Get(0) == zero {
							if diftru.Get(0) > abnorm*ulp {
								result.Set(3, ulpinv)
							}
						} else if diftru.Get(0) == zero {
							if dif.Get(0) > abnorm*ulp {
								result.Set(3, ulpinv)
							}
						} else if dif.Get(4) == zero {
							if diftru.Get(4) > abnorm*ulp {
								result.Set(3, ulpinv)
							}
						} else if diftru.Get(4) == zero {
							if dif.Get(4) > abnorm*ulp {
								result.Set(3, ulpinv)
							}
						} else {
							ratio1 = maxf64(math.Abs(diftru.Get(0)/dif.Get(0)), math.Abs(dif.Get(0)/diftru.Get(0)))
							ratio2 = maxf64(math.Abs(diftru.Get(4)/dif.Get(4)), math.Abs(dif.Get(4)/diftru.Get(4)))
							result.Set(3, maxf64(ratio1, ratio2))
						}

						ntestt = ntestt + 4

						//                    Print out tests which fail.
						for j = 1; j <= 4; j++ {
							if (result.Get(j-1) >= thrsh2 && j >= 4) || (result.Get(j-1) >= (*thresh) && j <= 3) {
								t.Fail()
								//                       If this is the first test to fail,
								//                       print a header to the data file.
								if nerrs == 0 {
									fmt.Printf("\n %3s -- Complex Expert Eigenvalue/vector problem driver\n", "ZXV")

									//                          Print out messages for built-in examples
									//
									//                          Matrix types
									fmt.Printf(" Matrix types: \n\n")
									fmt.Printf(" TYPE 1: Da is diagonal, Db is identity, \n     A = Y^(-H) Da X^(-1), B = Y^(-H) Db X^(-1) \n     YH and X are left and right eigenvectors. \n\n")
									fmt.Printf(" TYPE 2: Da is quasi-diagonal, Db is identity, \n     A = Y^(-H) Da X^(-1), B = Y^(-H) Db X^(-1) \n     YH and X are left and right eigenvectors. \n\n")

									//                          Tests performed
									fmt.Printf("\n Tests performed:  \n     a is alpha, b is beta, l is a left eigenvector, \n     r is a right eigenvector and %s means %s.\n 1 = maxint | ( b A - a B )%s l | / const.\n 2 = maxint | ( b A - a B ) r | / const.\n 3 = maxint ( Sest/Stru, Stru/Sest )  over all eigenvalues\n 4 = maxint( DIFest/DIFtru, DIFtru/DIFest )  over the 1st and 5th eigenvectors\n\n", "'", "transpose", "'")

								}
								nerrs = nerrs + 1
								if result.Get(j-1) < 10000.0 {
									fmt.Printf(" Type=%2d, IWA=%2d, IWB=%2d, IWX=%2d, IWY=%2d, result %2d is %8.2f\n", iptype, iwa, iwb, iwx, iwy, j, result.Get(j-1))
								} else {
									fmt.Printf(" Type=%2d, IWA=%2d, IWB=%2d, IWX=%2d, IWY=%2d, result %2d is %10.3E\n", iptype, iwa, iwb, iwx, iwy, j, result.Get(j-1))
								}
							}
						}

					label30:
					}
				}
			}
		}
	}

	// goto label150

	nlist := []int{4, 4}
	alist := [][]complex128{
		{
			2.0000e00 + 6.0000e00i, 2.0000e00 + 5.0000e00i, 3.0000e00 + -1.0000e01i, 4.0000e00 + 7.0000e00i,
			0.0000e00 + 0.0000e00i, 9.0000e00 + 2.0000e00i, 1.6000e01 + -2.4000e01i, 7.0000e00 + -7.0000e00i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 8.0000e00 + -3.0000e00i, 9.0000e00 + -8.0000e00i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 1.0000e01 + -1.6000e01i,
		},
		{
			1.0000e00 + 8.0000e00i, 2.0000e00 + 4.0000e00i, 3.0000e00 + -1.3000e01i, 4.0000e00 + 4.0000e00i,
			0.0000e00 + 0.0000e00i, 5.0000e00 + 7.0000e00i, 6.0000e00 + -2.4000e01i, 7.0000e00 + -3.0000e00i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 8.0000e00 + 3.0000e00i, 9.0000e00 + -5.0000e00i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 1.0000e01 + 1.6000e01i,
		},
	}
	blist := [][]complex128{
		{
			-9.0000e00 + 1.0000e00i, -1.0000e00 + -8.0000e00i, -1.0000e00 + 1.0000e01i, 2.0000e00 + -6.0000e00i,
			0.0000e00 + 0.0000e00i, -1.0000e00 + 4.0000e00i, 1.0000e00 + 1.6000e01i, -6.0000e00 + 4.0000e00i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 1.0000e00 + -1.4000e01i, -1.0000e00 + 6.0000e00i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 8.0000e00 + 4.0000e00i,
		},
		{
			-1.0000e00 + 9.0000e00i, -1.0000e00 + -1.0000e00i, -1.0000e00 + 1.0000e00i, -1.0000e00 + -6.0000e00i,
			0.0000e00 + 0.0000e00i, -1.0000e00 + 4.0000e00i, -1.0000e00 + 1.6000e01i, -1.0000e00 + -2.4000e01i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 1.0000e00 + -1.1000e01i, -1.0000e00 + 6.0000e00i,
			0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 0.0000e00 + 0.0000e00i, 1.0000e00 + 4.0000e00i,
		},
	}
	dtrulist := [][]float64{
		{5.2612e00, 8.0058e-01, 1.4032e00, 4.0073e00},
		{4.9068e00, 1.6813e00, 3.4636e00, 5.2436e00},
	}
	diftrulist := [][]float64{
		{1.1787e00, 3.3139e00, 1.1835e00, 2.0777e00},
		{1.0386e+00, 1.4728e+00, 2.0029e+00, 9.8365e-01},
	}

	//     Read in data from file to check accuracy of condition estimation
	//     Read input data until N=0
	for _i, n = range nlist {
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
				b.Set(i-1, j-1, blist[_i][(i-1)*(n)+j-1])
			}
			dtru.Set(i-1, dtrulist[_i][i-1])
			diftru.Set(i-1, diftrulist[_i][i-1])
		}

		nptknt = nptknt + 1

		//     Compute eigenvalues/eigenvectors of (A, B).
		//     Compute eigenvalue/eigenvector condition numbers
		//     using computed eigenvectors.
		golapack.Zlacpy('F', &n, &n, a, lda, ai, lda)
		golapack.Zlacpy('F', &n, &n, b, lda, bi, lda)

		golapack.Zggevx('N', 'V', 'V', 'B', &n, ai, lda, bi, lda, alpha, beta, vl, lda, vr, lda, ilo, ihi, lscale, rscale, &anorm, &bnorm, s, dif, work, lwork, rwork, iwork, bwork, &linfo)

		if linfo != 0 {
			t.Fail()
			fmt.Printf(" ZDRGVX: %s returned INFO=%6d.\n         N=%6d, Input example #%2d)\n", "ZGGEVX", linfo, n, nptknt)
			goto label140
		}

		//     Compute the norm(A, B)
		golapack.Zlacpy('F', &n, &n, ai, lda, work.CMatrix(n, opts), &n)
		golapack.Zlacpy('F', &n, &n, bi, lda, work.CMatrixOff(n*n+1-1, n, opts), &n)
		abnorm = golapack.Zlange('F', &n, toPtr(2*n), work.CMatrix(n, opts), &n, rwork)

		//     Tests (1) and (2)
		result.Set(0, zero)
		Zget52(true, &n, a, lda, b, lda, vl, lda, alpha, beta, work, rwork, result.Off(0))
		if result.Get(1) > (*thresh) {
			t.Fail()
			fmt.Printf(" ZDRGVX: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, Input Example #%2d)\n", "Left", "ZGGEVX", result.Get(1), n, nptknt)
		}

		result.Set(1, zero)
		Zget52(false, &n, a, lda, b, lda, vr, lda, alpha, beta, work, rwork, result.Off(1))
		if result.Get(2) > (*thresh) {
			t.Fail()
			fmt.Printf(" ZDRGVX: %s Eigenvectors from %s incorrectly normalized.\n Bits of error=%10.3f,         N=%6d, Input Example #%2d)\n", "Right", "ZGGEVX", result.Get(2), n, nptknt)
		}

		//     Test (3)
		result.Set(2, zero)
		for i = 1; i <= n; i++ {
			if s.Get(i-1) == zero {
				if dtru.Get(i-1) > abnorm*ulp {
					result.Set(2, ulpinv)
				}
			} else if dtru.Get(i-1) == zero {
				if s.Get(i-1) > abnorm*ulp {
					result.Set(2, ulpinv)
				}
			} else {
				rwork.Set(i-1, maxf64(math.Abs(dtru.Get(i-1)/s.Get(i-1)), math.Abs(s.Get(i-1)/dtru.Get(i-1))))
				result.Set(2, maxf64(result.Get(2), rwork.Get(i-1)))
			}
		}

		//     Test (4)
		result.Set(3, zero)
		if dif.Get(0) == zero {
			if diftru.Get(0) > abnorm*ulp {
				result.Set(3, ulpinv)
			}
		} else if diftru.Get(0) == zero {
			if dif.Get(0) > abnorm*ulp {
				result.Set(3, ulpinv)
			}
		} else if dif.Get(4) == zero {
			if diftru.Get(4) > abnorm*ulp {
				result.Set(3, ulpinv)
			}
		} else if diftru.Get(4) == zero {
			if dif.Get(4) > abnorm*ulp {
				result.Set(3, ulpinv)
			}
		} else {
			ratio1 = maxf64(math.Abs(diftru.Get(0)/dif.Get(0)), math.Abs(dif.Get(0)/diftru.Get(0)))
			ratio2 = maxf64(math.Abs(diftru.Get(4)/dif.Get(4)), math.Abs(dif.Get(4)/diftru.Get(4)))
			result.Set(3, maxf64(ratio1, ratio2))
		}

		ntestt = ntestt + 4

		//     Print out tests which fail.
		for j = 1; j <= 4; j++ {
			if result.Get(j-1) >= thrsh2 {
				t.Fail()
				//           If this is the first test to fail,
				//           print a header to the data file.
				if nerrs == 0 {
					fmt.Printf("\n %3s -- Complex Expert Eigenvalue/vector problem driver\n", "ZXV")

					//              Print out messages for built-in examples
					//
					//              Matrix types
					fmt.Printf("Input Example\n")

					//              Tests performed
					fmt.Printf("\n Tests performed:  \n     a is alpha, b is beta, l is a left eigenvector, \n     r is a right eigenvector and %s means %s.\n 1 = maxint | ( b A - a B )%s l | / const.\n 2 = maxint | ( b A - a B ) r | / const.\n 3 = maxint ( Sest/Stru, Stru/Sest )  over all eigenvalues\n 4 = maxint( DIFest/DIFtru, DIFtru/DIFest )  over the 1st and 5th eigenvectors\n\n", "'", "transpose", "'")

				}
				nerrs = nerrs + 1
				if result.Get(j-1) < 10000.0 {
					fmt.Printf(" Input example #%2d, matrix order=%4d, result %2d is %8.2f\n", nptknt, n, j, result.Get(j-1))
				} else {
					fmt.Printf(" Input example #%2d, matrix order=%4d, result %2d is %10.3E\n", nptknt, n, j, result.Get(j-1))
				}
			}
		}

	label140:
	}
	// label150:
	// 	;

	//     Summary
	Alasvm([]byte("ZXV"), &nerrs, &ntestt, func() *int { y := 0; return &y }())

	work.SetRe(0, float64(maxwrk))
}
