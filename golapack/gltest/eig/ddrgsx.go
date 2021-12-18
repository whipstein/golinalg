package eig

import (
	"fmt"
	"math"
	"testing"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
	"github.com/whipstein/golinalg/util"
)

// ddrgsx checks the nonsymmetric generalized eigenvalue (Schur form)
// problem expert driver Dggesx.
//
// Dggesx factors A and B as Q S Z' and Q T Z', where ' means
// transpose, T is upper triangular, S is in generalized Schur form
// (block upper triangular, with 1x1 and 2x2 blocks on the diagonal,
// the 2x2 blocks corresponding to complex conjugate pairs of
// generalized eigenvalues), and Q and Z are orthogonal.  It also
// computes the generalized eigenvalues (alpha(1),beta(1)), ...,
// (alpha(n),beta(n)). Thus, w(j) = alpha(j)/beta(j) is a root of the
// characteristic equation
//
//     det( A - w(j) B ) = 0
//
// Optionally it also reorders the eigenvalues so that a selected
// cluster of eigenvalues appears in the leading diagonal block of the
// Schur forms; computes a reciprocal condition number for the average
// of the selected eigenvalues; and computes a reciprocal condition
// number for the right and left deflating subspaces corresponding to
// the selected eigenvalues.
//
// When ddrgsx is called with NSIZE > 0, five (5) types of built-in
// matrix pairs are used to test the routine Dggesx.
//
// When ddrgsx is called with NSIZE = 0, it reads in test matrix data
// to test Dggesx.
//
// For each matrix pair, the following tests will be performed and
// compared with the threshold THRESH except for the tests (7) and (9):
//
// (1)   | A - Q S Z' | / ( |A| n ulp )
//
// (2)   | B - Q T Z' | / ( |B| n ulp )
//
// (3)   | I - QQ' | / ( n ulp )
//
// (4)   | I - ZZ' | / ( n ulp )
//
// (5)   if A is in Schur form (i.e. quasi-triangular form)
//
// (6)   maximum over j of D(j)  where:
//
//       if alpha(j) is real:
//                     |alpha(j) - S(j,j)|        |beta(j) - T(j,j)|
//           D(j) = ------------------------ + -----------------------
//                  max(|alpha(j)|,|S(j,j)|)   max(|beta(j)|,|T(j,j)|)
//
//       if alpha(j) is complex:
//                                 | det( s S - w T ) |
//           D(j) = ---------------------------------------------------
//                  ulp max( s norm(S), |w| norm(T) )*norm( s S - w T )
//
//           and S and T are here the 2 x 2 diagonal blocks of S and T
//           corresponding to the j-th and j+1-th eigenvalues.
//
// (7)   if sorting worked and SDIM is the number of eigenvalues
//       which were selected.
//
// (8)   the estimated value DIF does not differ from the true values of
//       Difu and Difl more than a factor 10*THRESH. If the estimate DIF
//       equals zero the corresponding true values of Difu and Difl
//       should be less than EPS*norm(A, B). If the true value of Difu
//       and Difl equal zero, the estimate DIF should be less than
//       EPS*norm(A, B).
//
// (9)   If info = N+3 is returned by Dggesx, the reordering "failed"
//       and we check that DIF = PL = PR = 0 and that the true value of
//       Difu and Difl is < EPS*norm(A, B). We count the events when
//       info=N+3.
//
// For read-in test matrices, the above tests are run except that the
// exact value for DIF (and PL) is input data.  Additionally, there is
// one more test run for read-in test matrices:
//
// (10)  the estimated value PL does not differ from the true value of
//       PLTRU more than a factor THRESH. If the estimate PL equals
//       zero the corresponding true value of PLTRU should be less than
//       EPS*norm(A, B). If the true value of PLTRU equal zero, the
//       estimate PL should be less than EPS*norm(A, B).
//
// Note that for the built-in tests, a total of 10*NSIZE*(NSIZE-1)
// matrix pairs are generated and tested. NSIZE should be kept small.
//
// SVD (routine DGESVD) is used for computing the true value of DIF_u
// and DIF_l when testing the built-in test problems.
//
// Built-in Test Matrices
// ======================
//
// All built-in test matrices are the 2 by 2 block of triangular
// matrices
//
//          A = [ A11 A12 ]    and      B = [ B11 B12 ]
//              [     A22 ]                 [     B22 ]
//
// where for different type of A11 and A22 are given as the following.
// A12 and B12 are chosen so that the generalized Sylvester equation
//
//          A11*R - L*A22 = -A12
//          B11*R - L*B22 = -B12
//
// have prescribed solution R and L.
//
// Type 1:  A11 = J_m(1,-1) and A_22 = J_k(1-a,1).
//          B11 = I_m, B22 = I_k
//          where J_k(a,b) is the k-by-k Jordan block with ``a'' on
//          diagonal and ``b'' on superdiagonal.
//
// Type 2:  A11 = (a_ij) = ( 2(.5-sin(i)) ) and
//          B11 = (b_ij) = ( 2(.5-sin(ij)) ) for i=1,...,m, j=i,...,m
//          A22 = (a_ij) = ( 2(.5-sin(i+j)) ) and
//          B22 = (b_ij) = ( 2(.5-sin(ij)) ) for i=m+1,...,k, j=i,...,k
//
// Type 3:  A11, A22 and B11, B22 are chosen as for Type 2, but each
//          second diagonal block in A_11 and each third diagonal block
//          in A_22 are made as 2 by 2 blocks.
//
// Type 4:  A11 = ( 20(.5 - sin(ij)) ) and B22 = ( 2(.5 - sin(i+j)) )
//             for i=1,...,m,  j=1,...,m and
//          A22 = ( 20(.5 - sin(i+j)) ) and B22 = ( 2(.5 - sin(ij)) )
//             for i=m+1,...,k,  j=m+1,...,k
//
// Type 5:  (A,B) and have potentially close or common eigenvalues and
//          very large departure from block diagonality A_11 is chosen
//          as the m x m leading submatrix of A_1:
//                  |  1  b                            |
//                  | -b  1                            |
//                  |        1+d  b                    |
//                  |         -b 1+d                   |
//           A_1 =  |                  d  1            |
//                  |                 -1  d            |
//                  |                        -d  1     |
//                  |                        -1 -d     |
//                  |                               1  |
//          and A_22 is chosen as the k x k leading submatrix of A_2:
//                  | -1  b                            |
//                  | -b -1                            |
//                  |       1-d  b                     |
//                  |       -b  1-d                    |
//           A_2 =  |                 d 1+b            |
//                  |               -1-b d             |
//                  |                       -d  1+b    |
//                  |                      -1+b  -d    |
//                  |                              1-d |
//          and matrix B are chosen as identity matrices (see DLATM5).
func ddrgsx(nsize int, ncmax int, thresh float64, nin *util.Reader, nout int, a, b, ai, bi, z, q *mat.Matrix, alphar, alphai, beta *mat.Vector, c *mat.Matrix, s, work *mat.Vector, lwork int, iwork []int, liwork int, bwork []bool, t *testing.T) (nerrs, ntestt int, err error) {
	var ilabad bool
	var sense byte
	var abnrm, bignum, diftru, one, pltru, smlnum, temp1, temp2, ten, thrsh2, ulp, ulpinv, weight, zero float64
	var _i, bdspac, i, i1, ifunc, iinfo, j, linfo, maxwrk, minwrk, mm, mn2, nptknt, ntest, prtype, qba, qbb int

	difest := vf(2)
	pl := vf(2)
	result := vf(10)

	zero = 0.0
	one = 1.0
	ten = 1.0e+1

	m := &gltest.Common.Mn.M
	n := &gltest.Common.Mn.N
	mplusn := &gltest.Common.Mn.Mplusn
	k := &gltest.Common.Mn.K
	fs := &gltest.Common.Mn.Fs

	//     Check for errors
	if nsize < 0 {
		err = fmt.Errorf("nsize < 0: nsize=%v", nsize)
	} else if thresh < zero {
		err = fmt.Errorf("thresh < zero: thresh=%v", thresh)
	} else if nout <= 0 {
		err = fmt.Errorf("nout <= 0: nout=%v", nout)
	} else if a.Rows < 1 || a.Rows < nsize {
		err = fmt.Errorf("a.Rows < 1 || a.Rows < nsize: a.Rows=%v, nsize=%v", a.Rows, nsize)
	} else if c.Rows < 1 || c.Rows < nsize*nsize/2 {
		err = fmt.Errorf("c.Rows < 1 || c.Rows < nsize*nsize/2: c.Rows=%v, nsize=%v", c.Rows, nsize)
	} else if liwork < nsize+6 {
		err = fmt.Errorf("liwork < nsize+6: liwork=%v, nsize=%v", liwork, nsize)
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	minwrk = 1
	if err == nil && lwork >= 1 {
		minwrk = max(10*(nsize+1), 5*nsize*nsize/2)

		//        workspace for sggesx
		maxwrk = 9*(nsize+1) + nsize*ilaenv(1, "Dgeqrf", []byte{' '}, nsize, 1, nsize, 0)
		maxwrk = max(maxwrk, 9*(nsize+1)+nsize*ilaenv(1, "Dorgqr", []byte{' '}, nsize, 1, nsize, -1))

		//        workspace for dgesvd
		bdspac = 5 * nsize * nsize / 2
		maxwrk = max(maxwrk, 3*nsize*nsize/2+nsize*nsize*ilaenv(1, "Dgebrd", []byte{' '}, nsize*nsize/2, nsize*nsize/2, -1, -1))
		maxwrk = max(maxwrk, bdspac)

		maxwrk = max(maxwrk, minwrk)

		work.Set(0, float64(maxwrk))
	}

	if lwork < minwrk {
		err = fmt.Errorf("lwork < minwrk: lwork=%v, minwrk=%v", lwork, minwrk)
	}

	if err != nil {
		gltest.Xerbla2("ddrgsx", err)
		return
	}

	//     Important constants
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	smlnum = golapack.Dlamch(SafeMinimum) / ulp
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)
	thrsh2 = ten * thresh
	ntestt = 0
	nerrs = 0

	//     Go to the tests for read-in matrix pairs
	ifunc = 0

	//     Test the built-in matrix pairs.
	//     Loop over different functions (IFUNC) of Dggesx, types (PRTYPE)
	//     of test matrices, different size (M+N)
	prtype = 0
	qba = 3
	qbb = 4
	weight = math.Sqrt(ulp)

	for ifunc = 0; ifunc <= 3; ifunc++ {
		for prtype = 1; prtype <= 5; prtype++ {
			for (*m) = 1; (*m) <= nsize-1; (*m)++ {
				for (*n) = 1; (*n) <= nsize-(*m); (*n)++ {

					weight = one / weight
					(*mplusn) = (*m) + (*n)

					//                 Generate test matrices
					(*fs) = true
					(*k) = 0

					golapack.Dlaset(Full, *mplusn, *mplusn, zero, zero, ai)
					golapack.Dlaset(Full, *mplusn, *mplusn, zero, zero, bi)

					matgen.Dlatm5(prtype, *m, *n, ai, ai.Off(*m, *m), ai.Off(0, *m), bi, bi.Off(*m, *m), bi.Off(0, *m), q, z, weight, qba, qbb)

					//                 Compute the Schur factorization and swapping the
					//                 m-by-m (1,1)-blocks with n-by-n (2,2)-blocks.
					//                 Swapping is accomplished via the function Dlctsx
					//                 which is supplied below.
					if ifunc == 0 {
						sense = 'N'
					} else if ifunc == 1 {
						sense = 'E'
					} else if ifunc == 2 {
						sense = 'V'
					} else if ifunc == 3 {
						sense = 'B'
					}

					golapack.Dlacpy(Full, *mplusn, *mplusn, ai, a)
					golapack.Dlacpy(Full, *mplusn, *mplusn, bi, b)

					if mm, linfo, err = golapack.Dggesx('V', 'V', 'S', dlctsx, sense, *mplusn, ai, bi, alphar, alphai, beta, q, z, pl, difest, work, lwork, &iwork, liwork, &bwork); linfo != 0 && linfo != (*mplusn)+2 || err != nil {
						result.Set(0, ulpinv)
						fmt.Printf(" ddrgsx: %s returned info=%6d.\n         n=%6d, jtype=%6d)\n", "Dggesx", linfo, *mplusn, prtype)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
						goto label30
					}

					//                 Compute the norm(A, B)
					golapack.Dlacpy(Full, *mplusn, *mplusn, ai, work.Matrix(*mplusn, opts))
					golapack.Dlacpy(Full, *mplusn, *mplusn, bi, work.Off((*mplusn)*(*mplusn)).Matrix(*mplusn, opts))
					abnrm = golapack.Dlange('F', *mplusn, 2*(*mplusn), work.Matrix(*mplusn, opts), work)

					//                 Do tests (1) to (4)
					result.Set(0, dget51(1, *mplusn, a, ai, q, z, work))
					result.Set(1, dget51(1, *mplusn, b, bi, q, z, work))
					result.Set(2, dget51(3, *mplusn, b, bi, q, q, work))
					result.Set(3, dget51(3, *mplusn, b, bi, z, z, work))
					ntest = 4

					//                 Do tests (5) and (6): check Schur form of A and
					//                 compare eigenvalues with diagonals.
					temp1 = zero
					result.Set(4, zero)
					result.Set(5, zero)

					for j = 1; j <= (*mplusn); j++ {
						ilabad = false
						if alphai.Get(j-1) == zero {
							temp2 = (math.Abs(alphar.Get(j-1)-ai.Get(j-1, j-1))/math.Max(smlnum, math.Max(math.Abs(alphar.Get(j-1)), math.Abs(ai.Get(j-1, j-1)))) + math.Abs(beta.Get(j-1)-bi.Get(j-1, j-1))/math.Max(smlnum, math.Max(math.Abs(beta.Get(j-1)), math.Abs(bi.Get(j-1, j-1))))) / ulp
							if j < (*mplusn) {
								if ai.Get(j, j-1) != zero {
									ilabad = true
									result.Set(4, ulpinv)
								}
							}
							if j > 1 {
								if ai.Get(j-1, j-1-1) != zero {
									ilabad = true
									result.Set(4, ulpinv)
								}
							}
						} else {
							if alphai.Get(j-1) > zero {
								i1 = j
							} else {
								i1 = j - 1
							}
							if i1 <= 0 || i1 >= (*mplusn) {
								ilabad = true
							} else if i1 < (*mplusn)-1 {
								if ai.Get(i1+2-1, i1) != zero {
									ilabad = true
									result.Set(4, ulpinv)
								}
							} else if i1 > 1 {
								if ai.Get(i1-1, i1-1-1) != zero {
									ilabad = true
									result.Set(4, ulpinv)
								}
							}
							if !ilabad {
								temp2, iinfo = dget53(ai.Off(i1-1, i1-1), bi.Off(i1-1, i1-1), beta.Get(j-1), alphar.Get(j-1), alphai.Get(j-1))
								if iinfo >= 3 {
									t.Fail()
									fmt.Printf(" ddrgsx: DGET53 returned info=%1d for eigenvalue %6d.\n         n=%6d, jtype=%6d)\n", iinfo, j, *mplusn, prtype)
									err = fmt.Errorf("iinfo=%v", abs(iinfo))
								}
							} else {
								temp2 = ulpinv
							}
						}
						temp1 = math.Max(temp1, temp2)
						if ilabad {
							t.Fail()
							fmt.Printf(" ddrgsx: S not in Schur form at eigenvalue %6d.\n         n=%6d, jtype=%6d)\n", j, *mplusn, prtype)
						}
					}
					result.Set(5, temp1)
					ntest = ntest + 2

					//                 Test (7) (if sorting worked)
					result.Set(6, zero)
					if linfo == (*mplusn)+3 {
						result.Set(6, ulpinv)
					} else if mm != (*n) {
						result.Set(6, ulpinv)
					}
					ntest = ntest + 1

					//                 Test (8): compare the estimated value DIF and its
					//                 value. first, compute the exact DIF.
					result.Set(7, zero)
					mn2 = mm * ((*mplusn) - mm) * 2
					if ifunc >= 2 && mn2 <= ncmax*ncmax {
						//                    Note: for either following two causes, there are
						//                    almost same number of test cases fail the test.
						matgen.Dlakf2(mm, (*mplusn)-mm, ai, ai.Off(mm, mm), bi, bi.Off(mm, mm), c)

						if _, err = golapack.Dgesvd('N', 'N', mn2, mn2, c, s, work.Matrix(1, opts), work.Off(1).Matrix(1, opts), work.Off(2), lwork-2); err != nil {
							panic(err)
						}
						diftru = s.Get(mn2 - 1)

						if difest.Get(1) == zero {
							if diftru > abnrm*ulp {
								result.Set(7, ulpinv)
							}
						} else if diftru == zero {
							if difest.Get(1) > abnrm*ulp {
								result.Set(7, ulpinv)
							}
						} else if (diftru > thrsh2*difest.Get(1)) || (diftru*thrsh2 < difest.Get(1)) {
							result.Set(7, math.Max(diftru/difest.Get(1), difest.Get(1)/diftru))
						}
						ntest = ntest + 1
					}

					//                 Test (9)
					result.Set(8, zero)
					if linfo == ((*mplusn) + 2) {
						if diftru > abnrm*ulp {
							result.Set(8, ulpinv)
						}
						if (ifunc > 1) && (difest.Get(1) != zero) {
							result.Set(8, ulpinv)
						}
						if (ifunc == 1) && (pl.Get(0) != zero) {
							result.Set(8, ulpinv)
						}
						ntest = ntest + 1
					}

					ntestt = ntestt + ntest

					//                 Print out tests which fail.
					for j = 1; j <= 9; j++ {
						if result.Get(j-1) >= thresh {
							t.Fail()
							//                       If this is the first test to fail,
							//                       print a header to the data file.
							if nerrs == 0 {
								fmt.Printf("\n %3s -- Real Expert Generalized Schur form problem driver\n", "DGX")

								//                          Matrix types
								fmt.Printf(" Matrix types: \n  1:  A is a block diagonal matrix of Jordan blocks and B is the identity \n      matrix, \n  2:  A and B are upper triangular matrices, \n  3:  A and B are as type 2, but each second diagonal block in A_11 and \n      each third diaongal block in A_22 are 2x2 blocks,\n  4:  A and B are block diagonal matrices, \n  5:  (A,B) has potentially close or common eigenvalues.\n\n")

								//                          Tests performed
								fmt.Printf("\n Tests performed:  (S is Schur, T is triangular, Q and Z are %s,\n                    a is alpha, b is beta, and %s means %s.)\n  1 = | A - Q S Z%s | / ( |A| n ulp )      2 = | B - Q T Z%s | / ( |B| n ulp )\n  3 = | I - QQ%s | / ( n ulp )             4 = | I - ZZ%s | / ( n ulp )\n  5 = 1/ULP  if A is not in Schur form S\n  6 = difference between (alpha,beta) and diagonals of (S,T)\n  7 = 1/ULP  if SDIM is not the correct number of selected eigenvalues\n  8 = 1/ULP  if DIFEST/DIFTRU > 10*THRESH or DIFTRU/DIFEST > 10*THRESH\n  9 = 1/ULP  if DIFEST <> 0 or DIFTRU > ULP*norm(A,B) when reordering fails\n 10 = 1/ULP  if PLEST/PLTRU > THRESH or PLTRU/PLEST > THRESH\n    ( Test 10 is only for input examples )\n\n", "orthogonal", "'", "transpose", "'", "'", "'", "'")

							}
							nerrs = nerrs + 1
							if result.Get(j-1) < 10000.0 {
								fmt.Printf(" Matrix order=%2d, type=%2d, a=%10.3E, order(A_11)=%2d, result %2d is %8.2f\n", *mplusn, prtype, weight, *m, j, result.Get(j-1))
							} else {
								fmt.Printf(" Matrix order=%2d, type=%2d, a=%10.3E, order(A_11)=%2d, result %2d is %10.3E\n", *mplusn, prtype, weight, *m, j, result.Get(j-1))
							}
						}
					}

				label30:
				}
			}
		}
	}

	mplusnlist := []int{4, 4}
	nlist := []int{2, 2}
	ailist := [][]float64{
		{
			8.0000e+00, 4.0000e+00, -1.3000e+01, 4.0000e+00,
			0.0000e+00, 7.0000e+00, -2.4000e+01, -3.0000e+00,
			0.0000e+00, 0.0000e+00, 3.0000e+00, -5.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6000e+01,
		},
		{
			1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00,
			0.0000e+00, 5.0000e+00, 6.0000e+00, 7.0000e+00,
			0.0000e+00, 0.0000e+00, 8.0000e+00, 9.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+01,
		},
	}
	bilist := [][]float64{
		{
			9.0000e+00, -1.0000e+00, 1.0000e+00, -6.0000e+00,
			0.0000e+00, 4.0000e+00, 1.6000e+01, -2.4000e+01,
			0.0000e+00, 0.0000e+00, -1.1000e+01, 6.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 4.0000e+00,
		},
		{
			-1.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00,
			0.0000e+00, -1.0000e+00, -1.0000e+00, -1.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e+00, -1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
		},
	}
	pltrulist := []float64{2.5901e-01, 9.8173e-01}
	diftrulist := []float64{1.7592e+00, 6.3649e-01}

	//     Read in data from file to check accuracy of condition estimation
	//     Read input data until n=0
	nptknt = 0
	for _i, *mplusn = range mplusnlist {
		*n = nlist[_i]
		for i = 1; i <= (*mplusn); i++ {
			for j = 1; j <= (*mplusn); j++ {
				ai.Set(i-1, j-1, ailist[_i][(i-1)*(*mplusn)+j-1])
				bi.Set(i-1, j-1, bilist[_i][(i-1)*(*mplusn)+j-1])
			}
		}
		pltru = pltrulist[_i]
		diftru = diftrulist[_i]

		nptknt = nptknt + 1
		(*fs) = true
		(*k) = 0
		(*m) = (*mplusn) - (*n)

		golapack.Dlacpy(Full, *mplusn, *mplusn, ai, a)
		golapack.Dlacpy(Full, *mplusn, *mplusn, bi, b)

		//     Compute the Schur factorization while swapping the
		//     m-by-m (1,1)-blocks with n-by-n (2,2)-blocks.
		if mm, linfo, err = golapack.Dggesx('V', 'V', 'S', dlctsx, 'B', *mplusn, ai, bi, alphar, alphai, beta, q, z, pl, difest, work, lwork, &iwork, liwork, &bwork); linfo != 0 && linfo != (*mplusn)+2 || err != nil {
			t.Fail()
			result.Set(0, ulpinv)
			fmt.Printf(" ddrgsx: %s returned info=%6d.\n         n=%6d, Input Example #%2d)\n", "Dggesx", linfo, *mplusn, nptknt)
			continue
		}

		//     Compute the norm(A, B)
		//        (should this be norm of (A,B) or (AI,BI)?)
		golapack.Dlacpy(Full, *mplusn, *mplusn, ai, work.Matrix(*mplusn, opts))
		golapack.Dlacpy(Full, *mplusn, *mplusn, bi, work.Off((*mplusn)*(*mplusn)).Matrix(*mplusn, opts))
		abnrm = golapack.Dlange('F', *mplusn, 2*(*mplusn), work.Matrix(*mplusn, opts), work)

		//     Do tests (1) to (4)
		result.Set(0, dget51(1, *mplusn, a, ai, q, z, work))
		result.Set(1, dget51(1, *mplusn, b, bi, q, z, work))
		result.Set(2, dget51(3, *mplusn, b, bi, q, q, work))
		result.Set(3, dget51(3, *mplusn, b, bi, z, z, work))

		//     Do tests (5) and (6): check Schur form of A and compare
		//     eigenvalues with diagonals.
		ntest = 6
		temp1 = zero
		result.Set(4, zero)
		result.Set(5, zero)

		for j = 1; j <= (*mplusn); j++ {
			ilabad = false
			if alphai.Get(j-1) == zero {
				temp2 = (math.Abs(alphar.Get(j-1)-ai.Get(j-1, j-1))/math.Max(smlnum, math.Max(math.Abs(alphar.Get(j-1)), math.Abs(ai.Get(j-1, j-1)))) + math.Abs(beta.Get(j-1)-bi.Get(j-1, j-1))/math.Max(smlnum, math.Max(math.Abs(beta.Get(j-1)), math.Abs(bi.Get(j-1, j-1))))) / ulp
				if j < (*mplusn) {
					if ai.Get(j, j-1) != zero {
						ilabad = true
						result.Set(4, ulpinv)
					}
				}
				if j > 1 {
					if ai.Get(j-1, j-1-1) != zero {
						ilabad = true
						result.Set(4, ulpinv)
					}
				}
			} else {
				if alphai.Get(j-1) > zero {
					i1 = j
				} else {
					i1 = j - 1
				}
				if i1 <= 0 || i1 >= (*mplusn) {
					ilabad = true
				} else if i1 < (*mplusn)-1 {
					if ai.Get(i1+2-1, i1) != zero {
						ilabad = true
						result.Set(4, ulpinv)
					}
				} else if i1 > 1 {
					if ai.Get(i1-1, i1-1-1) != zero {
						ilabad = true
						result.Set(4, ulpinv)
					}
				}
				if !ilabad {
					temp2, iinfo = dget53(ai.Off(i1-1, i1-1), bi.Off(i1-1, i1-1), beta.Get(j-1), alphar.Get(j-1), alphai.Get(j-1))
					if iinfo >= 3 {
						t.Fail()
						fmt.Printf(" ddrgsx: DGET53 returned info=%1d for eigenvalue %6d.\n         n=%6d, jtype=%6d)\n", iinfo, j, *mplusn, nptknt)
						err = fmt.Errorf("iinfo=%v", abs(iinfo))
					}
				} else {
					temp2 = ulpinv
				}
			}
			temp1 = math.Max(temp1, temp2)
			if ilabad {
				t.Fail()
				fmt.Printf(" ddrgsx: S not in Schur form at eigenvalue %6d.\n         n=%6d, jtype=%6d)\n", j, *mplusn, nptknt)
			}
		}
		result.Set(5, temp1)

		//     Test (7) (if sorting worked)  <--------- need to be checked.
		ntest = 7
		result.Set(6, zero)
		if linfo == (*mplusn)+3 {
			result.Set(6, ulpinv)
		}

		//     Test (8): compare the estimated value of DIF and its true value.
		ntest = 8
		result.Set(7, zero)
		if difest.Get(1) == zero {
			if diftru > abnrm*ulp {
				result.Set(7, ulpinv)
			}
		} else if diftru == zero {
			if difest.Get(1) > abnrm*ulp {
				result.Set(7, ulpinv)
			}
		} else if (diftru > thrsh2*difest.Get(1)) || (diftru*thrsh2 < difest.Get(1)) {
			result.Set(7, math.Max(diftru/difest.Get(1), difest.Get(1)/diftru))
		}

		//     Test (9)
		ntest = 9
		result.Set(8, zero)
		if linfo == ((*mplusn) + 2) {
			if diftru > abnrm*ulp {
				result.Set(8, ulpinv)
			}
			if (ifunc > 1) && (difest.Get(1) != zero) {
				result.Set(8, ulpinv)
			}
			if (ifunc == 1) && (pl.Get(0) != zero) {
				result.Set(8, ulpinv)
			}
		}

		//     Test (10): compare the estimated value of PL and it true value.
		ntest = 10
		result.Set(9, zero)
		if pl.Get(0) == zero {
			if pltru > abnrm*ulp {
				result.Set(9, ulpinv)
			}
		} else if pltru == zero {
			if pl.Get(0) > abnrm*ulp {
				result.Set(9, ulpinv)
			}
		} else if (pltru > thresh*pl.Get(0)) || (pltru*thresh < pl.Get(0)) {
			result.Set(9, ulpinv)
		}

		ntestt = ntestt + ntest

		//     Print out tests which fail.
		for j = 1; j <= ntest; j++ {
			if result.Get(j-1) >= thresh {
				t.Fail()
				//           If this is the first test to fail,
				//           print a header to the data file.
				if nerrs == 0 {
					fmt.Printf("\n %3s -- Real Expert Generalized Schur form problem driver\n", "DGX")

					//              Matrix types
					fmt.Printf("Input Example\n")

					//              Tests performed
					fmt.Printf("\n Tests performed:  (S is Schur, T is triangular, Q and Z are %s,\n                    a is alpha, b is beta, and %s means %s.)\n  1 = | A - Q S Z%s | / ( |A| n ulp )      2 = | B - Q T Z%s | / ( |B| n ulp )\n  3 = | I - QQ%s | / ( n ulp )             4 = | I - ZZ%s | / ( n ulp )\n  5 = 1/ULP  if A is not in Schur form S\n  6 = difference between (alpha,beta) and diagonals of (S,T)\n  7 = 1/ULP  if SDIM is not the correct number of selected eigenvalues\n  8 = 1/ULP  if DIFEST/DIFTRU > 10*THRESH or DIFTRU/DIFEST > 10*THRESH\n  9 = 1/ULP  if DIFEST <> 0 or DIFTRU > ULP*norm(A,B) when reordering fails\n 10 = 1/ULP  if PLEST/PLTRU > THRESH or PLTRU/PLEST > THRESH\n    ( Test 10 is only for input examples )\n\n", "orthogonal", "'", "transpose", "'", "'", "'", "'")

				}
				nerrs = nerrs + 1
				if result.Get(j-1) < 10000.0 {
					fmt.Printf(" Input example #%2d, matrix order=%4d, result %2d is %8.2f\n", nptknt, *mplusn, j, result.Get(j-1))
				} else {
					fmt.Printf(" Input example #%2d, matrix order=%4d, result %2d is %10.3E\n", nptknt, *mplusn, j, result.Get(j-1))
				}
			}

		}
	}

	//     Summary
	// alasvm("Dgx", nerrs, ntestt, 0)

	work.Set(0, float64(maxwrk))

	return
}
