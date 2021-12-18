package golapack

import (
	"fmt"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgbbrd reduces a complex general m-by-n band matrix A to real upper
// bidiagonal form B by a unitary transformation: Q**H * A * P = B.
//
// The routine computes B, and optionally forms Q or P**H, or computes
// Q**H*C for a given matrix C.
func Zgbbrd(vect byte, m, n, ncc, kl, ku int, ab *mat.CMatrix, d, e *mat.Vector, q, pt, c *mat.CMatrix, work *mat.CVector, rwork *mat.Vector) (err error) {
	var wantb, wantc, wantpt, wantq bool
	var cone, czero, ra, rb, rs, t complex128
	var abst, rc, zero float64
	var i, inca, j, j1, j2, kb, kb1, kk, klm, klu1, kun, l, minmn, ml, ml0, mu, mu0, nr, nrt int

	zero = 0.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	wantb = vect == 'B'
	wantq = vect == 'Q' || wantb
	wantpt = vect == 'P' || wantb
	wantc = ncc > 0
	klu1 = kl + ku + 1
	if !wantq && !wantpt && vect != 'N' {
		err = fmt.Errorf("!wantq && !wantpt && vect != 'N': vect='%c'", vect)
	} else if m < 0 {
		err = fmt.Errorf("m < 0:  m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ncc < 0 {
		err = fmt.Errorf("ncc < 0: ncc=%v", ncc)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku < 0: ku=%v", ku)
	} else if ab.Rows < klu1 {
		err = fmt.Errorf("ab.Rows < klu1: ab.Rows=%v, klu1=%v", ab.Rows, klu1)
	} else if q.Rows < 1 || wantq && q.Rows < max(1, m) {
		err = fmt.Errorf("q.Rows < 1 || wantq && q.Rows < max(1, m): vect='%c', q.Rows=%v, m=%v", vect, q.Rows, m)
	} else if pt.Rows < 1 || wantpt && pt.Rows < max(1, n) {
		err = fmt.Errorf("pt.Rows < 1 || wantpt && pt.Rows < max(1, n): vect='%c', pt.Rows=%v, n=%v", vect, pt.Rows, n)
	} else if c.Rows < 1 || wantc && c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < 1 || wantc && c.Rows < max(1, m): vect='%c', c.Rows=%v, m=%v", vect, c.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Zgbbrd", err)
		return
	}

	//     Initialize Q and P**H to the unit matrix, if needed
	if wantq {
		Zlaset(Full, m, m, czero, cone, q)
	}
	if wantpt {
		Zlaset(Full, n, n, czero, cone, pt)
	}

	//     Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	minmn = min(m, n)

	if kl+ku > 1 {
		//        Reduce to upper bidiagonal form if KU > 0; if KU = 0, reduce
		//        first to lower bidiagonal form and then transform to upper
		//        bidiagonal
		if ku > 0 {
			ml0 = 1
			mu0 = 2
		} else {
			ml0 = 2
			mu0 = 1
		}

		//        Wherever possible, plane rotations are generated and applied in
		//        vector operations of length NR over the index set J1:J2:KLU1.
		//
		//        The complex sines of the plane rotations are stored in WORK,
		//        and the real cosines in RWORK.
		klm = min(m-1, kl)
		kun = min(n-1, ku)
		kb = klm + kun
		kb1 = kb + 1
		inca = kb1 * ab.Rows
		nr = 0
		j1 = klm + 2
		j2 = 1 - kun

		for i = 1; i <= minmn; i++ {
			//           Reduce i-th column and i-th row of matrix to bidiagonal form
			ml = klm + 1
			mu = kun + 1
			for kk = 1; kk <= kb; kk++ {
				j1 = j1 + kb
				j2 = j2 + kb

				//              generate plane rotations to annihilate nonzero elements
				//              which have been created below the band
				if nr > 0 {
					Zlargv(nr, ab.Off(klu1-1, j1-klm-1-1).CVector(), inca, work.Off(j1-1), kb1, rwork.Off(j1-1), kb1)
				}

				//              apply plane rotations from the left
				for l = 1; l <= kb; l++ {
					if j2-klm+l-1 > n {
						nrt = nr - 1
					} else {
						nrt = nr
					}
					if nrt > 0 {
						Zlartv(nrt, ab.Off(klu1-l-1, j1-klm+l-1-1).CVector(), inca, ab.Off(klu1-l, j1-klm+l-1-1).CVector(), inca, rwork.Off(j1-1), work.Off(j1-1), kb1)
					}
				}

				if ml > ml0 {
					if ml <= m-i+1 {
						//                    generate plane rotation to annihilate a(i+ml-1,i)
						//                    within the band, and apply rotation from the left
						*rwork.GetPtr(i + ml - 1 - 1), *work.GetPtr(i + ml - 1 - 1), ra = Zlartg(ab.Get(ku+ml-1-1, i-1), ab.Get(ku+ml-1, i-1))
						ab.Set(ku+ml-1-1, i-1, ra)
						if i < n {
							Zrot(min(ku+ml-2, n-i), ab.Off(ku+ml-2-1, i).CVector(), ab.Rows-1, ab.Off(ku+ml-1-1, i).CVector(), ab.Rows-1, rwork.Get(i+ml-1-1), work.Get(i+ml-1-1))
						}
					}
					nr = nr + 1
					j1 = j1 - kb1
				}

				if wantq {
					//                 accumulate product of plane rotations in Q
					for j = j1; j <= j2; j += kb1 {
						Zrot(m, q.Off(0, j-1-1).CVector(), 1, q.Off(0, j-1).CVector(), 1, rwork.Get(j-1), work.GetConj(j-1))
					}
				}

				if wantc {
					//                 apply plane rotations to C
					for j = j1; j <= j2; j += kb1 {
						Zrot(ncc, c.Off(j-1-1, 0).CVector(), c.Rows, c.Off(j-1, 0).CVector(), c.Rows, rwork.Get(j-1), work.Get(j-1))
					}
				}

				if j2+kun > n {
					//                 adjust J2 to keep within the bounds of the matrix
					nr = nr - 1
					j2 = j2 - kb1
				}

				for j = j1; j <= j2; j += kb1 {
					//                 create nonzero element a(j-1,j+ku) above the band
					//                 and store it in WORK(n+1:2*n)
					work.Set(j+kun-1, work.Get(j-1)*ab.Get(0, j+kun-1))
					ab.Set(0, j+kun-1, rwork.GetCmplx(j-1)*ab.Get(0, j+kun-1))
				}

				//              generate plane rotations to annihilate nonzero elements
				//              which have been generated above the band
				if nr > 0 {
					Zlargv(nr, ab.Off(0, j1+kun-1-1).CVector(), inca, work.Off(j1+kun-1), kb1, rwork.Off(j1+kun-1), kb1)
				}

				//              apply plane rotations from the right
				for l = 1; l <= kb; l++ {
					if j2+l-1 > m {
						nrt = nr - 1
					} else {
						nrt = nr
					}
					if nrt > 0 {
						Zlartv(nrt, ab.Off(l, j1+kun-1-1).CVector(), inca, ab.Off(l-1, j1+kun-1).CVector(), inca, rwork.Off(j1+kun-1), work.Off(j1+kun-1), kb1)
					}
				}

				if ml == ml0 && mu > mu0 {
					if mu <= n-i+1 {
						//                    generate plane rotation to annihilate a(i,i+mu-1)
						//                    within the band, and apply rotation from the right
						*rwork.GetPtr(i + mu - 1 - 1), *work.GetPtr(i + mu - 1 - 1), ra = Zlartg(ab.Get(ku-mu+3-1, i+mu-2-1), ab.Get(ku-mu+2-1, i+mu-1-1))
						ab.Set(ku-mu+3-1, i+mu-2-1, ra)
						Zrot(min(kl+mu-2, m-i), ab.Off(ku-mu+4-1, i+mu-2-1).CVector(), 1, ab.Off(ku-mu+3-1, i+mu-1-1).CVector(), 1, rwork.Get(i+mu-1-1), work.Get(i+mu-1-1))
					}
					nr = nr + 1
					j1 = j1 - kb1
				}

				if wantpt {
					//                 accumulate product of plane rotations in P**H
					for j = j1; j <= j2; j += kb1 {
						Zrot(n, pt.Off(j+kun-1-1, 0).CVector(), pt.Rows, pt.Off(j+kun-1, 0).CVector(), pt.Rows, rwork.Get(j+kun-1), work.GetConj(j+kun-1))
					}
				}

				if j2+kb > m {
					//                 adjust J2 to keep within the bounds of the matrix
					nr = nr - 1
					j2 = j2 - kb1
				}

				for j = j1; j <= j2; j += kb1 {
					//                 create nonzero element a(j+kl+ku,j+ku-1) below the
					//                 band and store it in WORK(1:n)
					work.Set(j+kb-1, work.Get(j+kun-1)*ab.Get(klu1-1, j+kun-1))
					ab.Set(klu1-1, j+kun-1, rwork.GetCmplx(j+kun-1)*ab.Get(klu1-1, j+kun-1))
				}

				if ml > ml0 {
					ml = ml - 1
				} else {
					mu = mu - 1
				}
			}
		}
	}

	if ku == 0 && kl > 0 {
		//        A has been reduced to complex lower bidiagonal form
		//
		//        Transform lower bidiagonal form to upper bidiagonal by applying
		//        plane rotations from the left, overwriting superdiagonal
		//        elements on subdiagonal elements
		for i = 1; i <= min(m-1, n); i++ {
			rc, rs, ra = Zlartg(ab.Get(0, i-1), ab.Get(1, i-1))
			ab.Set(0, i-1, ra)
			if i < n {
				ab.Set(1, i-1, rs*ab.Get(0, i))
				ab.Set(0, i, toCmplx(rc)*ab.Get(0, i))
			}
			if wantq {
				Zrot(m, q.Off(0, i-1).CVector(), 1, q.Off(0, i).CVector(), 1, rc, cmplx.Conj(rs))
			}
			if wantc {
				Zrot(ncc, c.Off(i-1, 0).CVector(), c.Rows, c.Off(i, 0).CVector(), c.Rows, rc, rs)
			}
		}
	} else {
		//        A has been reduced to complex upper bidiagonal form or is
		//        diagonal
		if ku > 0 && m < n {
			//           Annihilate a(m,m+1) by applying plane rotations from the
			//           right
			rb = ab.Get(ku-1, m)
			for i = m; i >= 1; i-- {
				rc, rs, ra = Zlartg(ab.Get(ku, i-1), rb)
				ab.Set(ku, i-1, ra)
				if i > 1 {
					rb = -cmplx.Conj(rs) * ab.Get(ku-1, i-1)
					ab.Set(ku-1, i-1, toCmplx(rc)*ab.Get(ku-1, i-1))
				}
				if wantpt {
					Zrot(n, pt.Off(i-1, 0).CVector(), pt.Rows, pt.Off(m, 0).CVector(), pt.Rows, rc, cmplx.Conj(rs))
				}
			}
		}
	}

	//     Make diagonal and superdiagonal elements real, storing them in D
	//     and E
	t = ab.Get(ku, 0)
	for i = 1; i <= minmn; i++ {
		abst = cmplx.Abs(t)
		d.Set(i-1, abst)
		if abst != zero {
			t = t / toCmplx(abst)
		} else {
			t = cone
		}
		if wantq {
			q.Off(0, i-1).CVector().Scal(m, t, 1)
		}
		if wantc {
			c.Off(i-1, 0).CVector().Scal(ncc, cmplx.Conj(t), c.Rows)
		}
		if i < minmn {
			if ku == 0 && kl == 0 {
				e.Set(i-1, zero)
				t = ab.Get(0, i)
			} else {
				if ku == 0 {
					t = ab.Get(1, i-1) * cmplx.Conj(t)
				} else {
					t = ab.Get(ku-1, i) * cmplx.Conj(t)
				}
				abst = cmplx.Abs(t)
				e.Set(i-1, abst)
				if abst != zero {
					t = t / toCmplx(abst)
				} else {
					t = cone
				}
				if wantpt {
					pt.Off(i, 0).CVector().Scal(n, t, pt.Rows)
				}
				t = ab.Get(ku, i) * cmplx.Conj(t)
			}
		}
	}

	return
}
