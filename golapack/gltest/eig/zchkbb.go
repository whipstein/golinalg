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

// Zchkbb tests the reduction of a general complex rectangular band
// matrix to real bidiagonal form.
//
// ZGBBRD factors a general band matrix A as  Q B P* , where * means
// conjugate transpose, B is upper bidiagonal, and Q and P are unitary;
// ZGBBRD can also overwrite a given matrix C with Q* C .
//
// For each pair of matrix dimensions (M,N) and each selected matrix
// type, an M by N matrix A and an M by NRHS matrix C are generated.
// The problem dimensions are as follows
//    A:          M x N
//    Q:          M x M
//    P:          N x N
//    B:          min(M,N) x min(M,N)
//    C:          M x NRHS
//
// For each generated matrix, 4 tests are performed:
//
// (1)   | A - Q B PT | / ( |A| max(M,N) ulp ), PT = P'
//
// (2)   | I - Q' Q | / ( M ulp )
//
// (3)   | I - PT PT' | / ( N ulp )
//
// (4)   | Y - Q' C | / ( |Y| max(M,NRHS) ulp ), where Y = Q' C.
//
// The "types" are specified by a logical array DOTYPE( 1:NTYPES );
// if DOTYPE(j) is .TRUE., then matrix type "j" will be generated.
// Currently, the list of possible types is:
//
// The possible matrix types are
//
// (1)  The zero matrix.
// (2)  The identity matrix.
//
// (3)  A diagonal matrix with evenly spaced entries
//      1, ..., ULP  and random signs.
//      (ULP = (first number larger than 1) - 1 )
// (4)  A diagonal matrix with geometrically spaced entries
//      1, ..., ULP  and random signs.
// (5)  A diagonal matrix with "clustered" entries 1, ULP, ..., ULP
//      and random signs.
//
// (6)  Same as (3), but multiplied by SQRT( overflow threshold )
// (7)  Same as (3), but multiplied by SQRT( underflow threshold )
//
// (8)  A matrix of the form  U D V, where U and V are orthogonal and
//      D has evenly spaced entries 1, ..., ULP with random signs
//      on the diagonal.
//
// (9)  A matrix of the form  U D V, where U and V are orthogonal and
//      D has geometrically spaced entries 1, ..., ULP with random
//      signs on the diagonal.
//
// (10) A matrix of the form  U D V, where U and V are orthogonal and
//      D has "clustered" entries 1, ULP,..., ULP with random
//      signs on the diagonal.
//
// (11) Same as (8), but multiplied by SQRT( overflow threshold )
// (12) Same as (8), but multiplied by SQRT( underflow threshold )
//
// (13) Rectangular matrix with random entries chosen from (-1,1).
// (14) Same as (13), but multiplied by SQRT( overflow threshold )
// (15) Same as (13), but multiplied by SQRT( underflow threshold )
func Zchkbb(nsizes *int, mval *[]int, nval *[]int, nwdths *int, kk *[]int, ntypes *int, dotype *[]bool, nrhs *int, iseed *[]int, thresh *float64, nounit *int, a *mat.CMatrix, lda *int, ab *mat.CMatrix, ldab *int, bd, be *mat.Vector, q *mat.CMatrix, ldq *int, p *mat.CMatrix, ldp *int, c *mat.CMatrix, ldc *int, cc *mat.CMatrix, work *mat.CVector, lwork *int, rwork, result *mat.Vector, info *int, t *testing.T) {
	var badmm, badnn, badnnb bool
	var cone, czero complex128
	var amninv, anorm, cond, one, ovfl, rtovfl, rtunfl, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, itype, j, jcol, jr, jsize, jtype, jwidth, k, kl, kmax, ku, m, maxtyp, mmax, mnmax, mtypes, n, nerrs, nmats, nmax, ntest, ntestt int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kmagn := make([]int, 15)
	kmode := make([]int, 15)
	ktype := make([]int, 15)

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)
	zero = 0.0
	one = 1.0
	maxtyp = 15

	ktype[0], ktype[1], ktype[2], ktype[3], ktype[4], ktype[5], ktype[6], ktype[7], ktype[8], ktype[9], ktype[10], ktype[11], ktype[12], ktype[13], ktype[14] = 1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9
	kmagn[0], kmagn[1], kmagn[2], kmagn[3], kmagn[4], kmagn[5], kmagn[6], kmagn[7], kmagn[8], kmagn[9], kmagn[10], kmagn[11], kmagn[12], kmagn[13], kmagn[14] = 1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3
	kmode[0], kmode[1], kmode[2], kmode[3], kmode[4], kmode[5], kmode[6], kmode[7], kmode[8], kmode[9], kmode[10], kmode[11], kmode[12], kmode[13], kmode[14] = 0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0

	//     Check for errors
	ntestt = 0
	(*info) = 0

	//     Important constants
	badmm = false
	badnn = false
	mmax = 1
	nmax = 1
	mnmax = 1
	for j = 1; j <= (*nsizes); j++ {
		mmax = max(mmax, (*mval)[j-1])
		if (*mval)[j-1] < 0 {
			badmm = true
		}
		nmax = max(nmax, (*nval)[j-1])
		if (*nval)[j-1] < 0 {
			badnn = true
		}
		mnmax = max(mnmax, min((*mval)[j-1], (*nval)[j-1]))
	}

	badnnb = false
	kmax = 0
	for j = 1; j <= (*nwdths); j++ {
		kmax = max(kmax, (*kk)[j-1])
		if (*kk)[j-1] < 0 {
			badnnb = true
		}
	}

	//     Check for errors
	if (*nsizes) < 0 {
		(*info) = -1
	} else if badmm {
		(*info) = -2
	} else if badnn {
		(*info) = -3
	} else if (*nwdths) < 0 {
		(*info) = -4
	} else if badnnb {
		(*info) = -5
	} else if (*ntypes) < 0 {
		(*info) = -6
	} else if (*nrhs) < 0 {
		(*info) = -8
	} else if (*lda) < nmax {
		(*info) = -13
	} else if (*ldab) < 2*kmax+1 {
		(*info) = -15
	} else if (*ldq) < nmax {
		(*info) = -19
	} else if (*ldp) < nmax {
		(*info) = -21
	} else if (*ldc) < nmax {
		(*info) = -23
	} else if (max(*lda, nmax)+1)*nmax > (*lwork) {
		(*info) = -26
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZCHKBB"), -(*info))
		return
	}

	//     Quick return if possible
	if (*nsizes) == 0 || (*ntypes) == 0 || (*nwdths) == 0 {
		return
	}

	//     More Important constants
	unfl = golapack.Dlamch(SafeMinimum)
	ovfl = one / unfl
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	ulpinv = one / ulp
	rtunfl = math.Sqrt(unfl)
	rtovfl = math.Sqrt(ovfl)

	//     Loop over sizes, widths, types
	nerrs = 0
	nmats = 0

	for jsize = 1; jsize <= (*nsizes); jsize++ {
		m = (*mval)[jsize-1]
		n = (*nval)[jsize-1]
		amninv = one / float64(max(1, m, n))

		for jwidth = 1; jwidth <= (*nwdths); jwidth++ {
			k = (*kk)[jwidth-1]
			if k >= m && k >= n {
				goto label150
			}
			kl = max(0, min(m-1, k))
			ku = max(0, min(n-1, k))

			if (*nsizes) != 1 {
				mtypes = min(maxtyp, *ntypes)
			} else {
				mtypes = min(maxtyp+1, *ntypes)
			}

			for jtype = 1; jtype <= mtypes; jtype++ {
				if !(*dotype)[jtype-1] {
					goto label140
				}
				nmats = nmats + 1
				ntest = 0

				for j = 1; j <= 4; j++ {
					ioldsd[j-1] = (*iseed)[j-1]
				}

				//              Compute "A".
				//
				//              Control parameters:
				//
				//                  KMAGN  KMODE        KTYPE
				//              =1  O(1)   clustered 1  zero
				//              =2  large  clustered 2  identity
				//              =3  small  exponential  (none)
				//              =4         arithmetic   diagonal, (w/ singular values)
				//              =5         random log   (none)
				//              =6         random       nonhermitian, w/ singular values
				//              =7                      (none)
				//              =8                      (none)
				//              =9                      random nonhermitian
				if mtypes > maxtyp {
					goto label90
				}

				itype = ktype[jtype-1]
				imode = kmode[jtype-1]

				//              Compute norm
				switch kmagn[jtype-1] {
				case 1:
					goto label40
				case 2:
					goto label50
				case 3:
					goto label60
				}

			label40:
				;
				anorm = one
				goto label70

			label50:
				;
				anorm = (rtovfl * ulp) * amninv
				goto label70

			label60:
				;
				anorm = rtunfl * float64(max(m, n)) * ulpinv
				goto label70

			label70:
				;

				golapack.Zlaset('F', lda, &n, &czero, &czero, a, lda)
				golapack.Zlaset('F', ldab, &n, &czero, &czero, ab, ldab)
				iinfo = 0
				cond = ulpinv

				//              Special Matrices -- Identity & Jordan block
				//
				//                 Zero
				if itype == 1 {
					iinfo = 0

				} else if itype == 2 {
					//                 Identity
					for jcol = 1; jcol <= n; jcol++ {
						a.Set(jcol-1, jcol-1, toCmplx(anorm))
					}

				} else if itype == 4 {
					//                 Diagonal Matrix, singular values specified
					matgen.Zlatms(&m, &n, 'S', iseed, 'N', rwork, &imode, &cond, &anorm, func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), 'N', a, lda, work, &iinfo)

				} else if itype == 6 {
					//                 Nonhermitian, singular values specified
					matgen.Zlatms(&m, &n, 'S', iseed, 'N', rwork, &imode, &cond, &anorm, &kl, &ku, 'N', a, lda, work, &iinfo)

				} else if itype == 9 {
					//                 Nonhermitian, random entries
					matgen.Zlatmr(&m, &n, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(n), func() *int { y := 1; return &y }(), &one, work.Off(2*n), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &kl, &ku, &zero, &anorm, 'N', a, lda, &idumma, &iinfo)

				} else {

					iinfo = 1
				}

				//              Generate Right-Hand Side
				matgen.Zlatmr(&m, nrhs, 'S', iseed, 'N', work, func() *int { y := 6; return &y }(), &one, &cone, 'T', 'N', work.Off(m), func() *int { y := 1; return &y }(), &one, work.Off(2*m), func() *int { y := 1; return &y }(), &one, 'N', &idumma, &m, nrhs, &zero, &one, 'N', c, ldc, &idumma, &iinfo)

				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKBB: %s returned INFO=%5d.\n         M=%5d N=%5d K=%5d, JTYPE=%5d, ISEED=%5d\n", "Generator", iinfo, m, n, k, jtype, ioldsd)
					(*info) = abs(iinfo)
					return
				}

			label90:
				;

				//              Copy A to band storage.
				for j = 1; j <= n; j++ {
					for i = max(1, j-ku); i <= min(m, j+kl); i++ {
						ab.Set(ku+1+i-j-1, j-1, a.Get(i-1, j-1))
					}
				}

				//              Copy C
				golapack.Zlacpy('F', &m, nrhs, c, ldc, cc, ldc)

				//              Call ZGBBRD to compute B, Q and P, and to update C.
				golapack.Zgbbrd('B', &m, &n, nrhs, &kl, &ku, ab, ldab, bd, be, q, ldq, p, ldp, cc, ldc, work, rwork, &iinfo)

				if iinfo != 0 {
					t.Fail()
					fmt.Printf(" ZCHKBB: %s returned INFO=%5d.\n         M=%5d N=%5d K=%5d, JTYPE=%5d, ISEED=%5d\n", "ZGBBRD", iinfo, m, n, k, jtype, ioldsd)
					(*info) = abs(iinfo)
					if iinfo < 0 {
						return
					} else {
						result.Set(0, ulpinv)
						goto label120
					}
				}

				//              Test 1:  Check the decomposition A := Q * B * P'
				//                   2:  Check the orthogonality of Q
				//                   3:  Check the orthogonality of P
				//                   4:  Check the computation of Q' * C
				Zbdt01(&m, &n, toPtr(-1), a, lda, q, ldq, bd, be, p, ldp, work, rwork, result.GetPtr(0))
				Zunt01('C', &m, &m, q, ldq, work, lwork, rwork, result.GetPtr(1))
				Zunt01('R', &n, &n, p, ldp, work, lwork, rwork, result.GetPtr(2))
				Zbdt02(&m, nrhs, c, ldc, cc, ldc, q, ldq, work, rwork, result.GetPtr(3))

				//              End of Loop -- Check for RESULT(j) > THRESH
				ntest = 4
			label120:
				;
				ntestt = ntestt + ntest

				//              Print out tests which fail.
				for jr = 1; jr <= ntest; jr++ {
					if result.Get(jr-1) >= (*thresh) {
						t.Fail()
						if nerrs == 0 {
							Dlahd2([]byte("ZBB"))
						}
						nerrs = nerrs + 1
						fmt.Printf(" M =%4d N=%4d, K=%3d, seed=%4d, type %2d, test(%2d)=%10.3f\n", m, n, k, ioldsd, jtype, jr, result.Get(jr-1))
					}
				}

			label140:
			}
		label150:
		}
	}

	//     Summary
	Dlasum([]byte("ZBB"), &nerrs, &ntestt)
}
