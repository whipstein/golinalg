package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgbbrd reduces a real general m-by-n band matrix A to upper
// bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//
// The routine computes B, and optionally forms Q or P**T, or computes
// Q**T*C for a given matrix C.
func Dgbbrd(vect byte, m, n, ncc, kl, ku int, ab *mat.Matrix, d, e *mat.Vector, q, pt, c *mat.Matrix, work *mat.Vector) (err error) {
	var wantb, wantc, wantpt, wantq bool
	var one, ra, rb, rc, rs, zero float64
	var i, inca, j, j1, j2, kb, kb1, kk, klm, klu1, kun, l, minmn, ml, ml0, mn, mu, mu0, nr, nrt int

	zero = 0.0
	one = 1.0

	//     Test the input parameters
	wantb = vect == 'B'
	wantq = vect == 'Q' || wantb
	wantpt = vect == 'P' || wantb
	wantc = ncc > 0
	klu1 = kl + ku + 1
	if !wantq && !wantpt && vect != 'N' {
		err = fmt.Errorf("!wantq && !wantpt && vect != 'N': vect='%c'", vect)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ncc < 0 {
		err = fmt.Errorf("ncc < 0: ncc=%v", ncc)
	} else if kl < 0 {
		err = fmt.Errorf("kl < 0: kl=%v", kl)
	} else if ku < 0 {
		err = fmt.Errorf("ku < 0: ku='%c'", ku)
	} else if ab.Rows < klu1 {
		err = fmt.Errorf("ab.Rows < klu1: ab.Rows=%v, kl=%v, ku=%v", ab.Rows, kl, ku)
	} else if q.Rows < 1 || wantq && q.Rows < max(1, m) {
		err = fmt.Errorf("q.Rows < 1 || wantq && q.Rows < max(1, m): vect='%c', q.Rows=%v, m=%v", vect, q.Rows, m)
	} else if pt.Rows < 1 || wantpt && pt.Rows < max(1, n) {
		err = fmt.Errorf("pt.Rows < 1 || wantpt && pt.Rows < max(1, n): vect='%c', pt.Rows=%v, n=%v", vect, pt.Rows, n)
	} else if c.Rows < 1 || wantc && c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < 1 || wantc && c.Rows < max(1, m): vect=%v, c.Rows=%v, m=%v", vect, c.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Dgbrrd", err)
		return
	}

	//     Initialize Q and P**T to the unit matrix, if needed
	if wantq {
		Dlaset(Full, m, m, zero, one, q)
	}
	if wantpt {
		Dlaset(Full, n, n, zero, one, pt)
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
		//        The sines of the plane rotations are stored in WORK(1:max(m,n))
		//        and the cosines in WORK(max(m,n)+1:2*max(m,n)).
		mn = max(m, n)
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
					Dlargv(nr, ab.Off(klu1-1, j1-klm-1-1).Vector(), inca, work.Off(j1-1), kb1, work.Off(mn+j1-1), kb1)
				}

				//              apply plane rotations from the left
				for l = 1; l <= kb; l++ {
					if j2-klm+l-1 > n {
						nrt = nr - 1
					} else {
						nrt = nr
					}
					if nrt > 0 {
						Dlartv(nrt, ab.Off(klu1-l-1, j1-klm+l-1-1).Vector(), inca, ab.Off(klu1-l, j1-klm+l-1-1).Vector(), inca, work.Off(mn+j1-1), work.Off(j1-1), kb1)
					}
				}

				if ml > ml0 {
					if ml <= m-i+1 {
						//                    generate plane rotation to annihilate a(i+ml-1,i)
						//                    within the band, and apply rotation from the left
						*work.GetPtr(mn + i + ml - 1 - 1), *work.GetPtr(i + ml - 1 - 1), ra = Dlartg(ab.Get(ku+ml-1-1, i-1), ab.Get(ku+ml-1, i-1))
						ab.Set(ku+ml-1-1, i-1, ra)
						if i < n {
							ab.Off(ku+ml-1-1, i).Vector().Rot(min(ku+ml-2, n-i), ab.Off(ku+ml-2-1, i).Vector(), ab.Rows-1, ab.Rows-1, work.Get(mn+i+ml-1-1), work.Get(i+ml-1-1))
						}
					}
					nr = nr + 1
					j1 = j1 - kb1
				}

				if wantq {
					//                 accumulate product of plane rotations in Q
					for j = j1; j <= j2; j += kb1 {
						q.Off(0, j-1).Vector().Rot(m, q.Off(0, j-1-1).Vector(), 1, 1, work.Get(mn+j-1), work.Get(j-1))
					}
				}

				if wantc {
					//                 apply plane rotations to C
					for j = j1; j <= j2; j += kb1 {
						c.Off(j-1, 0).Vector().Rot(ncc, c.Off(j-1-1, 0).Vector(), c.Rows, c.Rows, work.Get(mn+j-1), work.Get(j-1))
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
					ab.Set(0, j+kun-1, work.Get(mn+j-1)*ab.Get(0, j+kun-1))
				}

				//              generate plane rotations to annihilate nonzero elements
				//              which have been generated above the band
				if nr > 0 {
					Dlargv(nr, ab.Off(0, j1+kun-1-1).Vector(), inca, work.Off(j1+kun-1), kb1, work.Off(mn+j1+kun-1), kb1)
				}

				//              apply plane rotations from the right
				for l = 1; l <= kb; l++ {
					if j2+l-1 > m {
						nrt = nr - 1
					} else {
						nrt = nr
					}
					if nrt > 0 {
						Dlartv(nrt, ab.Off(l, j1+kun-1-1).Vector(), inca, ab.Off(l-1, j1+kun-1).Vector(), inca, work.Off(mn+j1+kun-1), work.Off(j1+kun-1), kb1)
					}
				}

				if ml == ml0 && mu > mu0 {
					if mu <= n-i+1 {
						//                    generate plane rotation to annihilate a(i,i+mu-1)
						//                    within the band, and apply rotation from the right
						*work.GetPtr(mn + i + mu - 1 - 1), *work.GetPtr(i + mu - 1 - 1), ra = Dlartg(ab.Get(ku-mu+3-1, i+mu-2-1), ab.Get(ku-mu+2-1, i+mu-1-1))
						ab.Set(ku-mu+3-1, i+mu-2-1, ra)
						ab.Off(ku-mu+3-1, i+mu-1-1).Vector().Rot(min(kl+mu-2, m-i), ab.Off(ku-mu+4-1, i+mu-2-1).Vector(), 1, 1, work.Get(mn+i+mu-1-1), work.Get(i+mu-1-1))
					}
					nr = nr + 1
					j1 = j1 - kb1
				}

				if wantpt {
					//                 accumulate product of plane rotations in P**T
					for j = j1; j <= j2; j += kb1 {
						pt.Off(j+kun-1, 0).Vector().Rot(n, pt.Off(j+kun-1-1, 0).Vector(), pt.Rows, pt.Rows, work.Get(mn+j+kun-1), work.Get(j+kun-1))
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
					ab.Set(klu1-1, j+kun-1, work.Get(mn+j+kun-1)*ab.Get(klu1-1, j+kun-1))
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
		//        A has been reduced to lower bidiagonal form
		//
		//        Transform lower bidiagonal form to upper bidiagonal by applying
		//        plane rotations from the left, storing diagonal elements in D
		//        and off-diagonal elements in E
		for i = 1; i <= min(m-1, n); i++ {
			rc, rs, ra = Dlartg(ab.Get(0, i-1), ab.Get(1, i-1))
			d.Set(i-1, ra)
			if i < n {
				e.Set(i-1, rs*ab.Get(0, i))
				ab.Set(0, i, rc*ab.Get(0, i))
			}
			if wantq {
				q.Off(0, i).Vector().Rot(m, q.Off(0, i-1).Vector(), 1, 1, rc, rs)
			}
			if wantc {
				c.Off(i, 0).Vector().Rot(ncc, c.Off(i-1, 0).Vector(), c.Rows, c.Rows, rc, rs)
			}
		}
		if m <= n {
			d.Set(m-1, ab.Get(0, m-1))
		}
	} else if ku > 0 {
		//        A has been reduced to upper bidiagonal form
		if m < n {
			//           Annihilate a(m,m+1) by applying plane rotations from the
			//           right, storing diagonal elements in D and off-diagonal
			//           elements in E
			rb = ab.Get(ku-1, m)
			for i = m; i >= 1; i-- {
				rc, rs, ra = Dlartg(ab.Get(ku, i-1), rb)
				d.Set(i-1, ra)
				if i > 1 {
					rb = -rs * ab.Get(ku-1, i-1)
					e.Set(i-1-1, rc*ab.Get(ku-1, i-1))
				}
				if wantpt {
					pt.Off(m, 0).Vector().Rot(n, pt.Off(i-1, 0).Vector(), pt.Rows, pt.Rows, rc, rs)
				}
			}
		} else {
			//           Copy off-diagonal elements to E and diagonal elements to D
			for i = 1; i <= minmn-1; i++ {
				e.Set(i-1, ab.Get(ku-1, i))
			}
			for i = 1; i <= minmn; i++ {
				d.Set(i-1, ab.Get(ku, i-1))
			}
		}
	} else {
		//        A is diagonal. Set elements of E to zero and copy diagonal
		//        elements to D.
		for i = 1; i <= minmn-1; i++ {
			e.Set(i-1, zero)
		}
		for i = 1; i <= minmn; i++ {
			d.Set(i-1, ab.Get(0, i-1))
		}
	}

	return
}
