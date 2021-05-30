package eig

import (
	"fmt"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/golapack/gltest/matgen"
	"golinalg/mat"
	"math"
	"testing"
)

// Zdrgsx checks the nonsymmetric generalized eigenvalue (Schur form)
// problem expert driver ZGGESX.
//
// ZGGES factors A and B as Q*S*Z'  and Q*T*Z' , where ' means conjugate
// transpose, S and T are  upper triangular (i.e., in generalized Schur
// form), and Q and Z are unitary. It also computes the generalized
// eigenvalues (alpha(j),beta(j)), j=1,...,n.  Thus,
// w(j) = alpha(j)/beta(j) is a root of the characteristic equation
//
//                 det( A - w(j) B ) = 0
//
// Optionally it also reorders the eigenvalues so that a selected
// cluster of eigenvalues appears in the leading diagonal block of the
// Schur forms; computes a reciprocal condition number for the average
// of the selected eigenvalues; and computes a reciprocal condition
// number for the right and left deflating subspaces corresponding to
// the selected eigenvalues.
//
// When ZDRGSX is called with NSIZE > 0, five (5) types of built-in
// matrix pairs are used to test the routine ZGGESX.
//
// When ZDRGSX is called with NSIZE = 0, it reads in test matrix data
// to test ZGGESX.
// (need more details on what kind of read-in data are needed).
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
// (5)   if A is in Schur form (i.e. triangular form)
//
// (6)   maximum over j of D(j)  where:
//
//                     |alpha(j) - S(j,j)|        |beta(j) - T(j,j)|
//           D(j) = ------------------------ + -----------------------
//                  max(|alpha(j)|,|S(j,j)|)   max(|beta(j)|,|T(j,j)|)
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
// (9)   If INFO = N+3 is returned by ZGGESX, the reordering "failed"
//       and we check that DIF = PL = PR = 0 and that the true value of
//       Difu and Difl is < EPS*norm(A, B). We count the events when
//       INFO=N+3.
//
// For read-in test matrices, the same tests are run except that the
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
// SVD (routine ZGESVD) is used for computing the true value of DIF_u
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
func Zdrgsx(nsize, ncmax *int, thresh *float64, nout *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ai *mat.CMatrix, bi *mat.CMatrix, z *mat.CMatrix, q *mat.CMatrix, alpha *mat.CVector, beta *mat.CVector, c *mat.CMatrix, ldc *int, s *mat.Vector, work *mat.CVector, lwork *int, rwork *mat.Vector, iwork *[]int, liwork *int, bwork *[]bool, info *int, t *testing.T) {
	var ilabad bool
	var sense byte
	var czero complex128
	var abnrm, bignum, diftru, one, pltru, smlnum, temp1, temp2, ten, thrsh2, ulp, ulpinv, weight, zero float64
	var _i, bdspac, i, ifunc, j, linfo, maxwrk, minwrk, mm, mn2, nerrs, nptknt, ntest, ntestt, prtype, qba, qbb int
	var mplusnlist, nlist []int
	var ailist, bilist [][]complex128
	var pltrulist, diftrulist []float64
	difest := vf(2)
	pl := vf(2)
	result := vf(10)

	zero = 0.0
	one = 1.0
	ten = 1.0e+1
	czero = (0.0 + 0.0*1i)
	m := &gltest.Common.Mn.M
	n := &gltest.Common.Mn.N
	mplusn := &gltest.Common.Mn.Mplusn
	k := &gltest.Common.Mn.K
	fs := &gltest.Common.Mn.Fs

	//     Check for errors
	(*info) = 0
	if (*nsize) < 0 {
		(*info) = -1
	} else if (*thresh) < zero {
		(*info) = -2
	} else if (*nout) <= 0 {
		(*info) = -4
	} else if (*lda) < 1 || (*lda) < (*nsize) {
		(*info) = -6
	} else if (*ldc) < 1 || (*ldc) < (*nsize)*(*nsize)/2 {
		(*info) = -15
	} else if (*liwork) < (*nsize)+2 {
		(*info) = -21
	}

	//     Compute workspace
	//      (Note: Comments in the code beginning "Workspace:" describe the
	//       minimal amount of workspace needed at that point in the code,
	//       as well as the preferred amount for good performance.
	//       NB refers to the optimal block size for the immediately
	//       following subroutine, as returned by ILAENV.)
	minwrk = 1
	if (*info) == 0 && (*lwork) >= 1 {
		minwrk = 3 * (*nsize) * (*nsize) / 2

		//        workspace for cggesx
		maxwrk = (*nsize) * (1 + Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEQRF"), []byte{' '}, nsize, func() *int { y := 1; return &y }(), nsize, func() *int { y := 0; return &y }()))
		maxwrk = maxint(maxwrk, (*nsize)*(1+Ilaenv(func() *int { y := 1; return &y }(), []byte("ZUNGQR"), []byte{' '}, nsize, func() *int { y := 1; return &y }(), nsize, toPtr(-1))))

		//        workspace for zgesvd
		bdspac = 3 * (*nsize) * (*nsize) / 2
		maxwrk = maxint(maxwrk, (*nsize)*(*nsize)*(1+Ilaenv(func() *int { y := 1; return &y }(), []byte("ZGEBRD"), []byte{' '}, toPtr((*nsize)*(*nsize)/2), toPtr((*nsize)*(*nsize)/2), toPtr(-1), toPtr(-1))))
		maxwrk = maxint(maxwrk, bdspac)

		maxwrk = maxint(maxwrk, minwrk)

		work.SetRe(0, float64(maxwrk))
	}

	if (*lwork) < minwrk {
		(*info) = -18
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZDRGSX"), -(*info))
		return
	}

	//     Important constants
	ulp = golapack.Dlamch(Precision)
	ulpinv = one / ulp
	smlnum = golapack.Dlamch(SafeMinimum) / ulp
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)
	thrsh2 = ten * (*thresh)
	ntestt = 0
	nerrs = 0

	//     Go to the tests for read-in matrix pairs
	ifunc = 0
	if (*nsize) == 0 {
		goto label70
	}

	//     Test the built-in matrix pairs.
	//     Loop over different functions (IFUNC) of ZGGESX, types (PRTYPE)
	//     of test matrices, different size (M+N)
	prtype = 0
	qba = 3
	qbb = 4
	weight = math.Sqrt(ulp)

	for ifunc = 0; ifunc <= 3; ifunc++ {
		for prtype = 1; prtype <= 5; prtype++ {
			for (*m) = 1; (*m) <= (*nsize)-1; (*m)++ {
				for (*n) = 1; (*n) <= (*nsize)-(*m); (*n)++ {

					weight = one / weight
					(*mplusn) = (*m) + (*n)

					//                 Generate test matrices
					(*fs) = true
					(*k) = 0

					golapack.Zlaset('F', mplusn, mplusn, &czero, &czero, ai, lda)
					golapack.Zlaset('F', mplusn, mplusn, &czero, &czero, bi, lda)

					matgen.Zlatm5(&prtype, m, n, ai, lda, ai.Off((*m)+1-1, (*m)+1-1), lda, ai.Off(0, (*m)+1-1), lda, bi, lda, bi.Off((*m)+1-1, (*m)+1-1), lda, bi.Off(0, (*m)+1-1), lda, q, lda, z, lda, &weight, &qba, &qbb)

					//                 Compute the Schur factorization and swapping the
					//                 m-by-m (1,1)-blocks with n-by-n (2,2)-blocks.
					//                 Swapping is accomplished via the function Zlctsx
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

					golapack.Zlacpy('F', mplusn, mplusn, ai, lda, a, lda)
					golapack.Zlacpy('F', mplusn, mplusn, bi, lda, b, lda)

					golapack.Zggesx('V', 'V', 'S', Zlctsx, sense, mplusn, ai, lda, bi, lda, &mm, alpha, beta, q, lda, z, lda, pl, difest, work, lwork, rwork, iwork, liwork, bwork, &linfo)

					if linfo != 0 && linfo != (*mplusn)+2 {
						t.Fail()
						result.Set(0, ulpinv)
						fmt.Printf(" ZDRGSX: %s returned INFO=%6d.\n         N=%6d, JTYPE=%6d)\n", "ZGGESX", linfo, *mplusn, prtype)
						(*info) = linfo
						goto label30
					}

					//                 Compute the norm(A, B)
					golapack.Zlacpy('F', mplusn, mplusn, ai, lda, work.CMatrix(*mplusn, opts), mplusn)
					golapack.Zlacpy('F', mplusn, mplusn, bi, lda, work.CMatrixOff((*mplusn)*(*mplusn)+1-1, *mplusn, opts), mplusn)
					abnrm = golapack.Zlange('F', mplusn, toPtr(2*(*mplusn)), work.CMatrix(*mplusn, opts), mplusn, rwork)

					//                 Do tests (1) to (4)
					result.Set(1, zero)
					Zget51(func() *int { y := 1; return &y }(), mplusn, a, lda, ai, lda, q, lda, z, lda, work, rwork, result.GetPtr(0))
					Zget51(func() *int { y := 1; return &y }(), mplusn, b, lda, bi, lda, q, lda, z, lda, work, rwork, result.GetPtr(1))
					Zget51(func() *int { y := 3; return &y }(), mplusn, b, lda, bi, lda, q, lda, q, lda, work, rwork, result.GetPtr(2))
					Zget51(func() *int { y := 3; return &y }(), mplusn, b, lda, bi, lda, z, lda, z, lda, work, rwork, result.GetPtr(3))
					ntest = 4

					//                 Do tests (5) and (6): check Schur form of A and
					//                 compare eigenvalues with diagonals.
					temp1 = zero
					result.Set(4, zero)
					result.Set(5, zero)

					for j = 1; j <= (*mplusn); j++ {
						ilabad = false
						temp2 = (abs1(alpha.Get(j-1)-ai.Get(j-1, j-1))/maxf64(smlnum, abs1(alpha.Get(j-1)), abs1(ai.Get(j-1, j-1))) + abs1(beta.Get(j-1)-bi.Get(j-1, j-1))/maxf64(smlnum, abs1(beta.Get(j-1)), abs1(bi.Get(j-1, j-1)))) / ulp
						if j < (*mplusn) {
							if ai.GetRe(j+1-1, j-1) != zero {
								ilabad = true
								result.Set(4, ulpinv)
							}
						}
						if j > 1 {
							if ai.GetRe(j-1, j-1-1) != zero {
								ilabad = true
								result.Set(4, ulpinv)
							}
						}
						temp1 = maxf64(temp1, temp2)
						if ilabad {
							fmt.Printf(" ZDRGSX: S not in Schur form at eigenvalue %6d.\n         N=%6d, JTYPE=%6d)\n", j, *mplusn, prtype)
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
					if ifunc >= 2 && mn2 <= (*ncmax)*(*ncmax) {
						//                    Note: for either following two cases, there are
						//                    almost same number of test cases fail the test.
						matgen.Zlakf2(&mm, toPtr((*mplusn)-mm), ai, lda, ai.Off(mm+1-1, mm+1-1), bi, bi.Off(mm+1-1, mm+1-1), c, ldc)

						golapack.Zgesvd('N', 'N', &mn2, &mn2, c, ldc, s, work.CMatrix(1, opts), func() *int { y := 1; return &y }(), work.CMatrixOff(1, 1, opts), func() *int { y := 1; return &y }(), work.Off(2), toPtr((*lwork)-2), rwork, info)
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
							result.Set(7, maxf64(diftru/difest.Get(1), difest.Get(1)/diftru))
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
						if result.Get(j-1) >= (*thresh) {
							t.Fail()
							//                       If this is the first test to fail,
							//                       print a header to the data file.
							if nerrs == 0 {
								fmt.Printf("\n %3s -- Complex Expert Generalized Schur form problem driver\n", "ZGX")

								//                          Matrix types
								fmt.Printf(" Matrix types: \n  1:  A is a block diagonal matrix of Jordan blocks and B is the identity \n      matrix, \n  2:  A and B are upper triangular matrices, \n  3:  A and B are as type 2, but each second diagonal block in A_11 and \n      each third diaongal block in A_22 are 2x2 blocks,\n  4:  A and B are block diagonal matrices, \n  5:  (A,B) has potentially close or common eigenvalues.\n\n")

								//                          Tests performed
								fmt.Printf("\n Tests performed:  (S is Schur, T is triangular, Q and Z are %s,\n                    a is alpha, b is beta, and %s means %s.)\n  1 = | A - Q S Z%s | / ( |A| n ulp )      2 = | B - Q T Z%s | / ( |B| n ulp )\n  3 = | I - QQ%s | / ( n ulp )             4 = | I - ZZ%s | / ( n ulp )\n  5 = 1/ULP  if A is not in Schur form S\n  6 = difference between (alpha,beta) and diagonals of (S,T)\n  7 = 1/ULP  if SDIM is not the correct number of selected eigenvalues\n  8 = 1/ULP  if DIFEST/DIFTRU > 10*THRESH or DIFTRU/DIFEST > 10*THRESH\n  9 = 1/ULP  if DIFEST <> 0 or DIFTRU > ULP*norm(A,B) when reordering fails\n 10 = 1/ULP  if PLEST/PLTRU > THRESH or PLTRU/PLEST > THRESH\n    ( Test 10 is only for input examples )\n\n", "unitary", "'", "transpose", "'", "'", "'", "'")

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

	goto label150

label70:
	;

	//     Read in data from file to check accuracy of condition estimation
	//     Read input data until N=0
	nptknt = 0
	mplusnlist = []int{4, 4}
	nlist = []int{2, 2}
	ailist = [][]complex128{
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
	bilist = [][]complex128{
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
	pltrulist = []float64{7.6883e-02, 4.2067e-01}
	diftrulist = []float64{2.1007e-01, 4.9338e00}

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

		golapack.Zlacpy('F', mplusn, mplusn, ai, lda, a, lda)
		golapack.Zlacpy('F', mplusn, mplusn, bi, lda, b, lda)

		//     Compute the Schur factorization while swapping the
		//     m-by-m (1,1)-blocks with n-by-n (2,2)-blocks.
		golapack.Zggesx('V', 'V', 'S', Zlctsx, 'B', mplusn, ai, lda, bi, lda, &mm, alpha, beta, q, lda, z, lda, pl, difest, work, lwork, rwork, iwork, liwork, bwork, &linfo)

		if linfo != 0 && linfo != (*mplusn)+2 {
			t.Fail()
			result.Set(0, ulpinv)
			fmt.Printf(" ZDRGSX: %s returned INFO=%6d.\n         N=%6d, Input Example #%2d)\n", "ZGGESX", linfo, *mplusn, nptknt)
			continue
		}

		//     Compute the norm(A, B)
		//        (should this be norm of (A,B) or (AI,BI)?)
		golapack.Zlacpy('F', mplusn, mplusn, ai, lda, work.CMatrix(*mplusn, opts), mplusn)
		golapack.Zlacpy('F', mplusn, mplusn, bi, lda, work.CMatrixOff((*mplusn)*(*mplusn)+1-1, *mplusn, opts), mplusn)
		abnrm = golapack.Zlange('F', mplusn, toPtr(2*(*mplusn)), work.CMatrix(*mplusn, opts), mplusn, rwork)

		//     Do tests (1) to (4)
		Zget51(func() *int { y := 1; return &y }(), mplusn, a, lda, ai, lda, q, lda, z, lda, work, rwork, result.GetPtr(0))
		Zget51(func() *int { y := 1; return &y }(), mplusn, b, lda, bi, lda, q, lda, z, lda, work, rwork, result.GetPtr(1))
		Zget51(func() *int { y := 3; return &y }(), mplusn, b, lda, bi, lda, q, lda, q, lda, work, rwork, result.GetPtr(2))
		Zget51(func() *int { y := 3; return &y }(), mplusn, b, lda, bi, lda, z, lda, z, lda, work, rwork, result.GetPtr(3))

		//     Do tests (5) and (6): check Schur form of A and compare
		//     eigenvalues with diagonals.
		ntest = 6
		temp1 = zero
		result.Set(4, zero)
		result.Set(5, zero)

		for j = 1; j <= (*mplusn); j++ {
			ilabad = false
			temp2 = (abs1(alpha.Get(j-1)-ai.Get(j-1, j-1))/maxf64(smlnum, abs1(alpha.Get(j-1)), abs1(ai.Get(j-1, j-1))) + abs1(beta.Get(j-1)-bi.Get(j-1, j-1))/maxf64(smlnum, abs1(beta.Get(j-1)), abs1(bi.Get(j-1, j-1)))) / ulp
			if j < (*mplusn) {
				if ai.GetRe(j+1-1, j-1) != zero {
					ilabad = true
					result.Set(4, ulpinv)
				}
			}
			if j > 1 {
				if ai.GetRe(j-1, j-1-1) != zero {
					ilabad = true
					result.Set(4, ulpinv)
				}
			}
			temp1 = maxf64(temp1, temp2)
			if ilabad {
				fmt.Printf(" ZDRGSX: S not in Schur form at eigenvalue %6d.\n         N=%6d, JTYPE=%6d)\n", j, *mplusn, nptknt)
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
			result.Set(7, maxf64(diftru/difest.Get(1), difest.Get(1)/diftru))
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
		} else if (pltru > (*thresh)*pl.Get(0)) || (pltru*(*thresh) < pl.Get(0)) {
			result.Set(9, ulpinv)
		}

		ntestt = ntestt + ntest

		//     Print out tests which fail.
		for j = 1; j <= ntest; j++ {
			if result.Get(j-1) >= (*thresh) {
				t.Fail()
				//           If this is the first test to fail,
				//           print a header to the data file.
				if nerrs == 0 {
					fmt.Printf("\n %3s -- Complex Expert Generalized Schur form problem driver\n", "ZGX")

					//              Matrix types
					fmt.Printf("Input Example\n")

					//              Tests performed
					fmt.Printf("\n Tests performed:  (S is Schur, T is triangular, Q and Z are %s,\n                    a is alpha, b is beta, and %s means %s.)\n  1 = | A - Q S Z%s | / ( |A| n ulp )      2 = | B - Q T Z%s | / ( |B| n ulp )\n  3 = | I - QQ%s | / ( n ulp )             4 = | I - ZZ%s | / ( n ulp )\n  5 = 1/ULP  if A is not in Schur form S\n  6 = difference between (alpha,beta) and diagonals of (S,T)\n  7 = 1/ULP  if SDIM is not the correct number of selected eigenvalues\n  8 = 1/ULP  if DIFEST/DIFTRU > 10*THRESH or DIFTRU/DIFEST > 10*THRESH\n  9 = 1/ULP  if DIFEST <> 0 or DIFTRU > ULP*norm(A,B) when reordering fails\n 10 = 1/ULP  if PLEST/PLTRU > THRESH or PLTRU/PLEST > THRESH\n    ( Test 10 is only for input examples )\n\n", "unitary", "'", "transpose", "'", "'", "'", "'")

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

label150:
	;

	//     Summary
	Alasvm([]byte("ZGX"), &nerrs, &ntestt, func() *int { y := 0; return &y }())

	work.SetRe(0, float64(maxwrk))
}
