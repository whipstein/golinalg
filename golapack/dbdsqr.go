package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dbdsqr computes the singular values and, optionally, the right and/or
// left singular vectors from the singular value decomposition (SVD) of
// a real N-by-N (upper or lower) bidiagonal matrix B using the implicit
// zero-shift QR algorithm.  The SVD of B has the form
//
//    B = Q * S * P**T
//
// where S is the diagonal matrix of singular values, Q is an orthogonal
// matrix of left singular vectors, and P is an orthogonal matrix of
// right singular vectors.  If left singular vectors are requested, this
// subroutine actually returns U*Q instead of Q, and, if right singular
// vectors are requested, this subroutine returns P**T*VT instead of
// P**T, for given real input matrices U and VT.  When U and VT are the
// orthogonal matrices that reduce a general matrix A to bidiagonal
// form:  A = U*B*VT, as computed by DGEBRD, then
//
//    A = (U*Q) * S * (P**T*VT)
//
// is the SVD of A.  Optionally, the subroutine may also compute Q**T*C
// for a given real input matrix C.
//
// See "Computing  Small Singular Values of Bidiagonal Matrices With
// Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
// LAPACK Working Note #3 (or SIAM J. Sci. Statist. Comput. vol. 11,
// no. 5, pp. 873-912, Sept 1990) and
// "Accurate singular values and differential qd algorithms," by
// B. Parlett and V. Fernando, Technical Report CPAM-554, Mathematics
// Department, University of California at Berkeley, July 1992
// for a detailed description of the algorithm.
func Dbdsqr(uplo mat.MatUplo, n, ncvt, nru, ncc int, d, e *mat.Vector, vt *mat.Matrix, u *mat.Matrix, c *mat.Matrix, work *mat.Vector) (info int, err error) {
	var lower, rotate bool
	var abse, abss, cosl, cosr, cs, eps, f, g, h, hndrd, hndrth, meigth, mu, negone, oldcs, oldsn, one, r, shift, sigmn, sigmx, sinl, sinr, sll, smax, smin, sminl, sminoa, sn, ten, thresh, tol, tolmul, unfl, zero float64
	var i, idir, isub, iter, iterdivn, j, ll, lll, m, maxitdivn, maxitr, nm1, nm12, nm13, oldll, oldm int

	zero = 0.0
	one = 1.0
	negone = -1.0
	hndrth = 0.01
	ten = 10.0
	hndrd = 100.0
	meigth = -0.125
	maxitr = 6

	//     Test the input parameters.
	lower = uplo == Lower
	if uplo != Upper && !lower {
		err = fmt.Errorf("uplo != Upper && !lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ncvt < 0 {
		err = fmt.Errorf("ncvt < 0: ncvt=%v", ncvt)
	} else if nru < 0 {
		err = fmt.Errorf("nru < 0: nru=%v", nru)
	} else if ncc < 0 {
		err = fmt.Errorf("ncc < 0: ncc=%v", ncc)
	} else if (ncvt == 0 && vt.Rows < 1) || (ncvt > 0 && vt.Rows < max(1, n)) {
		err = fmt.Errorf("(ncvt == 0 && vt.Rows < 1) || (ncvt > 0 && vt.Rows < max(1, n)): vt.Rows=%v, n=%v, ncvt=%v", vt.Rows, n, ncvt)
	} else if u.Rows < max(1, nru) {
		err = fmt.Errorf("u.Rows < max(1, nru): u.Rows=%v, nru=%v", u.Rows, nru)
	} else if (ncc == 0 && c.Rows < 1) || (ncc > 0 && c.Rows < max(1, n)) {
		err = fmt.Errorf("(ncc == 0 && c.Rows < 1) || (ncc > 0 && c.Rows < max(1, n)): c.Rows=%v, n=%v, ncc=%v", c.Rows, n, ncc)
	}
	if err != nil {
		gltest.Xerbla2("Dbdsqr", err)
		return
	}

	if n == 0 {
		return
	}
	if n == 1 {
		goto label160
	}

	//     ROTATE is true if any singular vectors desired, false otherwise
	rotate = (ncvt > 0) || (nru > 0) || (ncc > 0)

	//     If no singular vectors desired, use qd algorithm
	if !rotate {
		if info, err = Dlasq1(n, d, e, work); err != nil {
			panic(err)
		}

		//     If INFO equals 2, dqds didn't finish, try to finish
		if info != 2 {
			return
		}
		info = 0
	}

	nm1 = n - 1
	nm12 = nm1 + nm1
	nm13 = nm12 + nm1
	idir = 0

	//     Get machine constants
	eps = Dlamch(Epsilon)
	unfl = Dlamch(SafeMinimum)

	//     If matrix lower bidiagonal, rotate to be upper bidiagonal
	//     by applying Givens rotations on the left
	if lower {
		for i = 1; i <= n-1; i++ {
			cs, sn, r = Dlartg(d.Get(i-1), e.Get(i-1))
			d.Set(i-1, r)
			e.Set(i-1, sn*d.Get(i))
			d.Set(i, cs*d.Get(i))
			work.Set(i-1, cs)
			work.Set(nm1+i-1, sn)
		}

		//        Update singular vectors if desired
		if nru > 0 {
			if err = Dlasr(Right, 'V', 'F', nru, n, work, work.Off(n-1), u); err != nil {
				panic(err)
			}
		}
		if ncc > 0 {
			if err = Dlasr(Left, 'V', 'F', n, ncc, work, work.Off(n-1), c); err != nil {
				panic(err)
			}
		}
	}

	//     Compute singular values to relative accuracy TOL
	//     (By setting TOL to be negative, algorithm will compute
	//     singular values to absolute accuracy ABS(TOL)*norm(input matrix))
	tolmul = math.Max(ten, math.Min(hndrd, math.Pow(eps, meigth)))
	tol = tolmul * eps

	//     Compute approximate maximum, minimum singular values
	smax = zero
	for i = 1; i <= n; i++ {
		smax = math.Max(smax, math.Abs(d.Get(i-1)))
	}
	for i = 1; i <= n-1; i++ {
		smax = math.Max(smax, math.Abs(e.Get(i-1)))
	}
	sminl = zero
	if tol >= zero {
		//        Relative accuracy desired
		sminoa = math.Abs(d.Get(0))
		if sminoa == zero {
			goto label50
		}
		mu = sminoa
		for i = 2; i <= n; i++ {
			mu = math.Abs(d.Get(i-1)) * (mu / (mu + math.Abs(e.Get(i-1-1))))
			sminoa = math.Min(sminoa, mu)
			if sminoa == zero {
				goto label50
			}
		}
	label50:
		;
		sminoa = sminoa / math.Sqrt(float64(n))
		thresh = math.Max(tol*sminoa, float64(maxitr)*(float64(n)*(float64(n)*unfl)))
	} else {
		//        Absolute accuracy desired
		thresh = math.Max(math.Abs(tol)*smax, float64(maxitr)*(float64(n)*(float64(n)*unfl)))
	}

	//     Prepare for main iteration loop for the singular values
	//     (MAXIT is the maximum number of passes through the inner
	//     loop permitted before nonconvergence signalled.)
	maxitdivn = maxitr * n
	iterdivn = 0
	iter = -1
	oldll = -1
	oldm = -1

	//     M points to last element of unconverged part of matrix
	m = n

	//     Begin main iteration loop
label60:
	;

	//     Check for convergence or exceeding iteration count
	if m <= 1 {
		goto label160
	}

	if iter >= n {
		iter = iter - n
		iterdivn = iterdivn + 1
		if iterdivn >= maxitdivn {
			goto label200
		}
	}

	//     Find diagonal block of matrix to work on
	if tol < zero && math.Abs(d.Get(m-1)) <= thresh {
		d.Set(m-1, zero)
	}
	smax = math.Abs(d.Get(m - 1))
	smin = smax
	for lll = 1; lll <= m-1; lll++ {
		ll = m - lll
		abss = math.Abs(d.Get(ll - 1))
		abse = math.Abs(e.Get(ll - 1))
		if tol < zero && abss <= thresh {
			d.Set(ll-1, zero)
		}
		if abse <= thresh {
			goto label80
		}
		smin = math.Min(smin, abss)
		smax = math.Max(smax, math.Max(abss, abse))
	}
	ll = 0
	goto label90
label80:
	;
	e.Set(ll-1, zero)

	//     Off splits since E(LL) = 0
	if ll == m-1 {
		//        Convergence of bottom singular value, return to top of loop
		m = m - 1
		goto label60
	}
label90:
	;
	ll = ll + 1

	//     E(LL) through E(M-1) are nonzero, E(LL-1) is zero
	if ll == m-1 {
		//        2 by 2 block, handle separately
		sigmn, sigmx, sinr, cosr, sinl, cosl = Dlasv2(d.Get(m-1-1), e.Get(m-1-1), d.Get(m-1))
		d.Set(m-1-1, sigmx)
		e.Set(m-1-1, zero)
		d.Set(m-1, sigmn)

		//        Compute singular vectors, if desired
		if ncvt > 0 {
			goblas.Drot(ncvt, vt.Vector(m-1-1, 0), vt.Vector(m-1, 0), cosr, sinr)
		}
		if nru > 0 {
			goblas.Drot(nru, u.Vector(0, m-1-1, 1), u.Vector(0, m-1, 1), cosl, sinl)
		}
		if ncc > 0 {
			goblas.Drot(ncc, c.Vector(m-1-1, 0), c.Vector(m-1, 0), cosl, sinl)
		}
		m = m - 2
		goto label60
	}

	//     If working on new submatrix, choose shift direction
	//     (from larger end diagonal element towards smaller)
	if ll > oldm || m < oldll {
		if math.Abs(d.Get(ll-1)) >= math.Abs(d.Get(m-1)) {
			//           Chase bulge from top (big end) to bottom (small end)
			idir = 1
		} else {
			//           Chase bulge from bottom (big end) to top (small end)
			idir = 2
		}
	}

	//     Apply convergence tests
	if idir == 1 {
		//        Run convergence test in forward direction
		//        First apply standard test to bottom of matrix
		if math.Abs(e.Get(m-1-1)) <= math.Abs(tol)*math.Abs(d.Get(m-1)) || (tol < zero && math.Abs(e.Get(m-1-1)) <= thresh) {
			e.Set(m-1-1, zero)
			goto label60
		}

		if tol >= zero {
			//           If relative accuracy desired,
			//           apply convergence criterion forward
			mu = math.Abs(d.Get(ll - 1))
			sminl = mu
			for lll = ll; lll <= m-1; lll++ {
				if math.Abs(e.Get(lll-1)) <= tol*mu {
					e.Set(lll-1, zero)
					goto label60
				}
				mu = math.Abs(d.Get(lll)) * (mu / (mu + math.Abs(e.Get(lll-1))))
				sminl = math.Min(sminl, mu)
			}
		}

	} else {
		//        Run convergence test in backward direction
		//        First apply standard test to top of matrix
		if math.Abs(e.Get(ll-1)) <= math.Abs(tol)*math.Abs(d.Get(ll-1)) || (tol < zero && math.Abs(e.Get(ll-1)) <= thresh) {
			e.Set(ll-1, zero)
			goto label60
		}

		if tol >= zero {
			//           If relative accuracy desired,
			//           apply convergence criterion backward
			mu = math.Abs(d.Get(m - 1))
			sminl = mu
			for lll = m - 1; lll >= ll; lll-- {
				if math.Abs(e.Get(lll-1)) <= tol*mu {
					e.Set(lll-1, zero)
					goto label60
				}
				mu = math.Abs(d.Get(lll-1)) * (mu / (mu + math.Abs(e.Get(lll-1))))
				sminl = math.Min(sminl, mu)
			}
		}
	}
	oldll = ll
	oldm = m

	//     Compute shift.  First, test if shifting would ruin relative
	//     accuracy, and if so set the shift to zero.
	if tol >= zero && float64(n)*tol*(sminl/smax) <= math.Max(eps, hndrth*tol) {
		//        Use a zero shift to avoid loss of relative accuracy
		shift = zero
	} else {
		//        Compute the shift from 2-by-2 block at end of matrix
		if idir == 1 {
			sll = math.Abs(d.Get(ll - 1))
			shift, r = Dlas2(d.Get(m-1-1), e.Get(m-1-1), d.Get(m-1))
		} else {
			sll = math.Abs(d.Get(m - 1))
			shift, r = Dlas2(d.Get(ll-1), e.Get(ll-1), d.Get(ll))
		}

		//        Test if shift negligible, and if so set to zero
		if sll > zero {
			if math.Pow(shift/sll, 2) < eps {
				shift = zero
			}
		}
	}

	//     Increment iteration count
	iter = iter + m - ll

	//     If SHIFT = 0, do simplified QR iteration
	if shift == zero {
		if idir == 1 {
			//           Chase bulge from top to bottom
			//           Save cosines and sines for later singular vector updates
			cs = one
			oldcs = one
			for i = ll; i <= m-1; i++ {
				cs, sn, r = Dlartg(d.Get(i-1)*cs, e.Get(i-1))
				if i > ll {
					e.Set(i-1-1, oldsn*r)
				}
				oldcs, oldsn, *d.GetPtr(i - 1) = Dlartg(oldcs*r, d.Get(i)*sn)
				work.Set(i-ll, cs)
				work.Set(i-ll+1+nm1-1, sn)
				work.Set(i-ll+1+nm12-1, oldcs)
				work.Set(i-ll+1+nm13-1, oldsn)
			}
			h = d.Get(m-1) * cs
			d.Set(m-1, h*oldcs)
			e.Set(m-1-1, h*oldsn)

			//           Update singular vectors
			if ncvt > 0 {
				if err = Dlasr(Left, 'V', 'F', m-ll+1, ncvt, work, work.Off(n-1), vt.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}
			if nru > 0 {
				if err = Dlasr(Right, 'V', 'F', nru, m-ll+1, work.Off(nm12), work.Off(nm13), u.Off(0, ll-1)); err != nil {
					panic(err)
				}
			}
			if ncc > 0 {
				if err = Dlasr(Left, 'V', 'F', m-ll+1, ncc, work.Off(nm12), work.Off(nm13), c.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}

			//           Test convergence
			if math.Abs(e.Get(m-1-1)) <= thresh {
				e.Set(m-1-1, zero)
			}

		} else {
			//           Chase bulge from bottom to top
			//           Save cosines and sines for later singular vector updates
			cs = one
			oldcs = one
			for i = m; i >= ll+1; i-- {
				cs, sn, r = Dlartg(d.Get(i-1)*cs, e.Get(i-1-1))
				if i < m {
					e.Set(i-1, oldsn*r)
				}
				oldcs, oldsn, *d.GetPtr(i - 1) = Dlartg(oldcs*r, d.Get(i-1-1)*sn)
				work.Set(i-ll-1, cs)
				work.Set(i-ll+nm1-1, -sn)
				work.Set(i-ll+nm12-1, oldcs)
				work.Set(i-ll+nm13-1, -oldsn)
			}
			h = d.Get(ll-1) * cs
			d.Set(ll-1, h*oldcs)
			e.Set(ll-1, h*oldsn)

			//           Update singular vectors
			if ncvt > 0 {
				if err = Dlasr(Left, 'V', 'B', m-ll+1, ncvt, work.Off(nm12), work.Off(nm13), vt.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}
			if nru > 0 {
				if err = Dlasr(Right, 'V', 'B', nru, m-ll+1, work, work.Off(n-1), u.Off(0, ll-1)); err != nil {
					panic(err)
				}
			}
			if ncc > 0 {
				if err = Dlasr(Left, 'V', 'B', m-ll+1, ncc, work, work.Off(n-1), c.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}

			//           Test convergence
			if math.Abs(e.Get(ll-1)) <= thresh {
				e.Set(ll-1, zero)
			}
		}
	} else {
		//        Use nonzero shift
		if idir == 1 {
			//           Chase bulge from top to bottom
			//           Save cosines and sines for later singular vector updates
			f = (math.Abs(d.Get(ll-1)) - shift) * (math.Copysign(one, d.Get(ll-1)) + shift/d.Get(ll-1))
			g = e.Get(ll - 1)
			for i = ll; i <= m-1; i++ {
				cosr, sinr, r = Dlartg(f, g)
				if i > ll {
					e.Set(i-1-1, r)
				}
				f = cosr*d.Get(i-1) + sinr*e.Get(i-1)
				e.Set(i-1, cosr*e.Get(i-1)-sinr*d.Get(i-1))
				g = sinr * d.Get(i)
				d.Set(i, cosr*d.Get(i))
				cosl, sinl, r = Dlartg(f, g)
				d.Set(i-1, r)
				f = cosl*e.Get(i-1) + sinl*d.Get(i)
				d.Set(i, cosl*d.Get(i)-sinl*e.Get(i-1))
				if i < m-1 {
					g = sinl * e.Get(i)
					e.Set(i, cosl*e.Get(i))
				}
				work.Set(i-ll, cosr)
				work.Set(i-ll+1+nm1-1, sinr)
				work.Set(i-ll+1+nm12-1, cosl)
				work.Set(i-ll+1+nm13-1, sinl)
			}
			e.Set(m-1-1, f)

			//           Update singular vectors
			if ncvt > 0 {
				if err = Dlasr(Left, 'V', 'F', m-ll+1, ncvt, work, work.Off(n-1), vt.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}
			if nru > 0 {
				if err = Dlasr(Right, 'V', 'F', nru, m-ll+1, work.Off(nm12), work.Off(nm13), u.Off(0, ll-1)); err != nil {
					panic(err)
				}
			}
			if ncc > 0 {
				if err = Dlasr(Left, 'V', 'F', m-ll+1, ncc, work.Off(nm12), work.Off(nm13), c.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}

			//           Test convergence
			if math.Abs(e.Get(m-1-1)) <= thresh {
				e.Set(m-1-1, zero)
			}

		} else {
			//           Chase bulge from bottom to top
			//           Save cosines and sines for later singular vector updates
			f = (math.Abs(d.Get(m-1)) - shift) * (math.Copysign(one, d.Get(m-1)) + shift/d.Get(m-1))
			g = e.Get(m - 1 - 1)
			for i = m; i >= ll+1; i-- {
				cosr, sinr, r = Dlartg(f, g)
				if i < m {
					e.Set(i-1, r)
				}
				f = cosr*d.Get(i-1) + sinr*e.Get(i-1-1)
				e.Set(i-1-1, cosr*e.Get(i-1-1)-sinr*d.Get(i-1))
				g = sinr * d.Get(i-1-1)
				d.Set(i-1-1, cosr*d.Get(i-1-1))
				cosl, sinl, r = Dlartg(f, g)
				d.Set(i-1, r)
				f = cosl*e.Get(i-1-1) + sinl*d.Get(i-1-1)
				d.Set(i-1-1, cosl*d.Get(i-1-1)-sinl*e.Get(i-1-1))
				if i > ll+1 {
					g = sinl * e.Get(i-2-1)
					e.Set(i-2-1, cosl*e.Get((i-2-1)))
				}
				work.Set(i-ll-1, cosr)
				work.Set(i-ll+nm1-1, -sinr)
				work.Set(i-ll+nm12-1, cosl)
				work.Set(i-ll+nm13-1, -sinl)
			}
			e.Set(ll-1, f)

			//           Test convergence
			if math.Abs(e.Get(ll-1)) <= thresh {
				e.Set(ll-1, zero)
			}

			//           Update singular vectors if desired
			if ncvt > 0 {
				if err = Dlasr(Left, 'V', 'B', m-ll+1, ncvt, work.Off(nm12), work.Off(nm13), vt.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}
			if nru > 0 {
				if err = Dlasr(Right, 'V', 'B', nru, m-ll+1, work, work.Off(n-1), u.Off(0, ll-1)); err != nil {
					panic(err)
				}
			}
			if ncc > 0 {
				if err = Dlasr(Left, 'V', 'B', m-ll+1, ncc, work, work.Off(n-1), c.Off(ll-1, 0)); err != nil {
					panic(err)
				}
			}
		}
	}

	//     QR iteration finished, go back and check convergence
	goto label60

	//     All singular values converged, so make them positive
label160:
	;
	for i = 1; i <= n; i++ {
		if d.Get(i-1) < zero {
			d.Set(i-1, -d.Get(i-1))

			//           Change sign of singular vectors, if desired
			if ncvt > 0 {
				// goblas.Dscal(*ncvt, negone, vt.Vector(i-1, 0, *&vt.Rows))
				goblas.Dscal(ncvt, negone, vt.Vector(i-1, 0))
			}
		}
	}

	//     Sort the singular values into decreasing order (insertion sort on
	//     singular values, but only one transposition per singular vector)
	for i = 1; i <= n-1; i++ {
		//        Scan for smallest D(I)
		isub = 1
		smin = d.Get(0)
		for j = 2; j <= n+1-i; j++ {
			if d.Get(j-1) <= smin {
				isub = j
				smin = d.Get(j - 1)
			}
		}
		if isub != n+1-i {
			//           Swap singular values and vectors
			d.Set(isub-1, d.Get(n+1-i-1))
			d.Set(n+1-i-1, smin)
			if ncvt > 0 {
				goblas.Dswap(ncvt, vt.Vector(isub-1, 0), vt.Vector(n+1-i-1, 0))
			}
			if nru > 0 {
				goblas.Dswap(nru, u.Vector(0, isub-1, 1), u.Vector(0, n+1-i-1, 1))
			}
			if ncc > 0 {
				goblas.Dswap(ncc, c.Vector(isub-1, 0), c.Vector(n+1-i-1, 0))
			}
		}
	}
	return

	//     Maximum number of iterations exceeded, failure to converge
label200:
	;
	info = 0
	for i = 1; i <= n-1; i++ {
		if e.Get(i-1) != zero {
			info = info + 1
		}
	}

	return
}
