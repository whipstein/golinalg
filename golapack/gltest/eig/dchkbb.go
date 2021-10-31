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

// dchkbb tests the reduction of a general real rectangular band
// matrix to bidiagonal form.
//
// Dgbbrd factors a general band matrix A as  Q B P* , where * means
// transpose, B is upper bidiagonal, and Q and P are orthogonal;
// Dgbbrd can also overwrite a given matrix C with Q* C .
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
func dchkbb(nsizes int, mval, nval []int, nwdths int, kk []int, ntypes int, dotype []bool, nrhs int, iseed *[]int, thresh float64, nounit int, a, ab *mat.Matrix, bd, be *mat.Vector, q, p, c, cc *mat.Matrix, work *mat.Vector, lwork int, result *mat.Vector, t *testing.T) (nerrs, ntestt int, err error) {
	var badmm, badnn, badnnb bool
	var amninv, anorm, cond, one, ovfl, rtovfl, rtunfl, ulp, ulpinv, unfl, zero float64
	var i, iinfo, imode, itype, j, jcol, jr, jsize, jtype, jwidth, k, kl, kmax, ku, m, maxtyp, mmax, mnmax, mtypes, n, nmats, nmax, ntest int
	idumma := make([]int, 1)
	ioldsd := make([]int, 4)
	kmagn := []int{1, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 2, 3}
	kmode := []int{0, 0, 4, 3, 1, 4, 4, 4, 3, 1, 4, 4, 0, 0, 0}
	ktype := []int{1, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 9, 9, 9}

	zero = 0.0
	one = 1.0
	maxtyp = 15

	//     Check for errors
	ntestt = 0

	//     Important constants
	badmm = false
	badnn = false
	mmax = 1
	nmax = 1
	mnmax = 1
	for j = 1; j <= nsizes; j++ {
		mmax = max(mmax, mval[j-1])
		if mval[j-1] < 0 {
			badmm = true
		}
		nmax = max(nmax, nval[j-1])
		if nval[j-1] < 0 {
			badnn = true
		}
		mnmax = max(mnmax, min(mval[j-1], nval[j-1]))
	}

	badnnb = false
	kmax = 0
	for j = 1; j <= nwdths; j++ {
		kmax = max(kmax, kk[j-1])
		if kk[j-1] < 0 {
			badnnb = true
		}
	}

	//     Check for errors
	if nsizes < 0 {
		err = fmt.Errorf("nsizes < 0: nsize=%v", nsizes)
	} else if badmm {
		err = fmt.Errorf("badmm: mval=%v", mval)
	} else if badnn {
		err = fmt.Errorf("badnn: nn=%v", nval)
	} else if nwdths < 0 {
		err = fmt.Errorf("nwdths < 0: nwdths=%v", nwdths)
	} else if badnnb {
		err = fmt.Errorf("badnnb: kk=%v", kk)
	} else if ntypes < 0 {
		err = fmt.Errorf("ntypes < 0: ntypes=%v", ntypes)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < nmax {
		err = fmt.Errorf("a.Rows < nmax: a.Rows=%v, nmax=%v", a.Rows, nmax)
	} else if ab.Rows < 2*kmax+1 {
		err = fmt.Errorf("ab.Rows < 2*kmax+1: ab.Rows=%v, kmax=%v", ab.Rows, kmax)
	} else if q.Rows < nmax {
		err = fmt.Errorf("q.Rows < nmax: q.Rows=%v, nmax=%v", q.Rows, nmax)
	} else if p.Rows < nmax {
		err = fmt.Errorf("p.Rows < nmax: p.Rows=%v, nmax=%v", p.Rows, nmax)
	} else if c.Rows < nmax {
		err = fmt.Errorf("c.Rows < nmax: c.Rows=%v, nmax=%v", c.Rows, nmax)
	} else if (max(a.Rows, nmax)+1)*nmax > lwork {
		err = fmt.Errorf("(max(a.Rows, nmax)+1)*nmax > lwork: a.Rows=%v, nmax=%v, lwork=%v", a.Rows, nmax, lwork)
	}

	if err != nil {
		gltest.Xerbla2("dchkbb", err)
		return
	}

	//     Quick return if possible
	if nsizes == 0 || ntypes == 0 || nwdths == 0 {
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

	for jsize = 1; jsize <= nsizes; jsize++ {
		m = mval[jsize-1]
		n = nval[jsize-1]
		amninv = one / float64(max(1, m, n))

		for jwidth = 1; jwidth <= nwdths; jwidth++ {
			k = kk[jwidth-1]
			if k >= m && k >= n {
				goto label150
			}
			kl = max(0, min(m-1, k))
			ku = max(0, min(n-1, k))

			if nsizes != 1 {
				mtypes = min(maxtyp, ntypes)
			} else {
				mtypes = min(maxtyp+1, ntypes)
			}

			for jtype = 1; jtype <= mtypes; jtype++ {
				if !dotype[jtype-1] {
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

				golapack.Dlaset(Full, a.Rows, n, zero, zero, a)
				golapack.Dlaset(Full, ab.Rows, n, zero, zero, ab)
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
						a.Set(jcol-1, jcol-1, anorm)
					}

				} else if itype == 4 {
					//                 Diagonal Matrix, singular values specified
					iinfo, err = matgen.Dlatms(m, n, 'S', iseed, 'N', work, imode, cond, anorm, 0, 0, 'N', a, work.Off(m))

				} else if itype == 6 {
					//                 Nonhermitian, singular values specified
					iinfo, err = matgen.Dlatms(m, n, 'S', iseed, 'N', work, imode, cond, anorm, kl, ku, 'N', a, work.Off(m))

				} else if itype == 9 {
					//                 Nonhermitian, random entries
					iinfo, err = matgen.Dlatmr(m, n, 'S', iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(n), 1, one, work.Off(2*n), 1, one, 'N', &idumma, kl, ku, zero, anorm, 'N', a, &idumma)

				} else {

					iinfo = 1
				}

				//              Generate Right-Hand Side
				iinfo, err = matgen.Dlatmr(m, nrhs, 'S', iseed, 'N', work, 6, one, one, 'T', 'N', work.Off(m), 1, one, work.Off(2*m), 1, one, 'N', &idumma, m, nrhs, zero, one, 'N', c, &idumma)

				if err != nil {
					t.Fail()
					fmt.Printf(" dchkbb: %s returned info=%5d.\n         m=%5d n=%5d k=%5d, jtype=%5d, iseed=%5d\n", "Generator", iinfo, m, n, k, jtype, ioldsd)
					err = fmt.Errorf("iinfo=%v", abs(iinfo))
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
				golapack.Dlacpy(Full, m, nrhs, c, cc)

				//              Call Dgbbrd to compute B, Q and P, and to update C.
				if err = golapack.Dgbbrd('B', m, n, nrhs, kl, ku, ab, bd, be, q, p, cc, work); err != nil {
					t.Fail()
					fmt.Printf(" dchkbb: %s returned info=%5d.\n         m=%5d n=%5d k=%5d, jtype=%5d, iseed=%5d\n", "Dgbbrd", iinfo, m, n, k, jtype, ioldsd)
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
				result.Set(0, dbdt01(m, n, -1, a, q, bd, be, p, work))
				result.Set(1, dort01('C', m, m, q, work, lwork))
				result.Set(2, dort01('R', n, n, p, work, lwork))
				result.Set(3, dbdt02(m, nrhs, c, cc, q, work))

				//              End of Loop -- Check for RESULT(j) > THRESH
				ntest = 4
			label120:
				;
				ntestt = ntestt + ntest

				//              Print out tests which fail.
				for jr = 1; jr <= ntest; jr++ {
					if result.Get(jr-1) >= thresh {
						t.Fail()
						if nerrs == 0 {
							dlahd2("Dbb")
						}
						nerrs = nerrs + 1
						fmt.Printf(" m=%4d n=%4d, k=%3d, seed=%4d, type %2d, test(%2d)=%10.3f\n", m, n, k, ioldsd, jtype, jr, result.Get(jr-1))
					}
				}

			label140:
			}
		label150:
		}
	}

	//     Summary
	// dlasum("Dbb", nerrs, ntestt)

	return
}
