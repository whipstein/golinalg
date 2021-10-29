package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dhseqr computes the eigenvalues of a Hessenberg matrix H
//    and, optionally, the matrices T and Z from the Schur decomposition
//    H = Z T Z**T, where T is an upper quasi-triangular matrix (the
//    Schur form), and Z is the orthogonal matrix of Schur vectors.
//
//    Optionally Z may be postmultiplied into an input orthogonal
//    matrix Q so that this routine can give the Schur factorization
//    of a matrix A which has been reduced to the Hessenberg form H
//    by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
func Dhseqr(job, compz byte, n, ilo, ihi int, h *mat.Matrix, wr, wi *mat.Vector, z *mat.Matrix, work *mat.Vector, lwork int) (info int, err error) {
	var initz, lquery, wantt, wantz bool
	var one, zero float64
	var i, kbot, nl, nmin, ntiny int

	ntiny = 11

	nl = 49
	zero = 0.0
	one = 1.0
	workl := vf(nl)
	hl := mf(nl, nl, opts)

	//     ==== Decode and check the input parameters. ====
	wantt = job == 'S'
	initz = compz == 'I'
	wantz = initz || compz == 'V'
	work.Set(0, float64(max(1, n)))
	lquery = lwork == -1

	if job != 'E' && !wantt {
		err = fmt.Errorf("job != 'E' && !wantt: job='%c'", job)
	} else if compz != 'N' && !wantz {
		err = fmt.Errorf("compz != 'N' && !wantz: compz='%c'", compz)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if ilo < 1 || ilo > max(1, n) {
		err = fmt.Errorf("ilo < 1 || ilo > max(1, n): n=%v, ilo=%v, ihi=%v", n, ilo, ihi)
	} else if ihi < min(ilo, n) || ihi > n {
		err = fmt.Errorf("ihi < min(ilo, n) || ihi > n: n=%v, ilo=%v, ihi=%v", n, ilo, ihi)
	} else if h.Rows < max(1, n) {
		err = fmt.Errorf("h.Rows < max(1, n): h.Rows=%v, n=%v", h.Rows, n)
	} else if z.Rows < 1 || (wantz && z.Rows < max(1, n)) {
		err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < max(1, n)): compz='%c', z.Rows=%v, n=%v", compz, z.Rows, n)
	} else if lwork < max(1, n) && !lquery {
		err = fmt.Errorf("lwork < max(1, n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	}

	if err != nil {
		//        ==== Quick return in case of invalid argument. ====
		gltest.Xerbla2("Dhseqr", err)
		return

	} else if n == 0 {
		//        ==== Quick return in case N = 0; nothing to do. ====
		return

	} else if lquery {
		//        ==== Quick return in case of a workspace query ====
		info = Dlaqr0(wantt, wantz, n, ilo, ihi, h, wr, wi, ilo, ihi, z, work, lwork)
		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, math.Max(float64(max(1, n)), work.Get(0)))
		return

	} else {
		//        ==== copy eigenvalues isolated by DGEBAL ====
		for i = 1; i <= ilo-1; i++ {
			wr.Set(i-1, h.Get(i-1, i-1))
			wi.Set(i-1, zero)
		}
		for i = ihi + 1; i <= n; i++ {
			wr.Set(i-1, h.Get(i-1, i-1))
			wi.Set(i-1, zero)
		}

		//        ==== Initialize Z, if requested ====
		if initz {
			Dlaset(Full, n, n, zero, one, z)
		}

		//        ==== Quick return if possible ====
		if ilo == ihi {
			wr.Set(ilo-1, h.Get(ilo-1, ilo-1))
			wi.Set(ilo-1, zero)
			return
		}

		//        ==== DLAHQR/DLAQR0 crossover point ====
		nmin = Ilaenv(12, "Dhseqr", []byte{job, compz}, n, ilo, ihi, lwork)
		nmin = max(ntiny, nmin)

		//        ==== DLAQR0 for big matrices; DLAHQR for small ones ====
		if n > nmin {
			info = Dlaqr0(wantt, wantz, n, ilo, ihi, h, wr, wi, ilo, ihi, z, work, lwork)
		} else {
			//           ==== Small matrix ====
			if info, err = Dlahqr(wantt, wantz, n, ilo, ihi, h, wr, wi, ilo, ihi, z); err != nil {
				panic(err)
			}

			if info > 0 {
				//              ==== A rare DLAHQR failure!  DLAQR0 sometimes succeeds
				//              .    when DLAHQR fails. ====
				kbot = info

				if n >= nl {
					//                 ==== Larger matrices have enough subdiagonal scratch
					//                 .    space to call DLAQR0 directly. ====
					info = Dlaqr0(wantt, wantz, n, ilo, kbot, h, wr, wi, ilo, ihi, z, work, lwork)

				} else {
					//                 ==== Tiny matrices don't have enough subdiagonal
					//                 .    scratch space to benefit from DLAQR0.  Hence,
					//                 .    tiny matrices must be copied into a larger
					//                 .    array before calling DLAQR0. ====
					Dlacpy(Full, n, n, h, hl)
					hl.Set(n, n-1, zero)
					Dlaset(Full, nl, nl-n, zero, zero, hl.Off(0, n))
					info = Dlaqr0(wantt, wantz, nl, ilo, kbot, hl, wr, wi, ilo, ihi, z, workl, nl)
					if wantt || info != 0 {
						Dlacpy(Full, n, n, hl, h)
					}
				}
			}
		}

		//        ==== Clear out the trash, if necessary. ====
		if (wantt || info != 0) && n > 2 {
			Dlaset(Lower, n-2, n-2, zero, zero, h.Off(2, 0))
		}

		//        ==== Ensure reported workspace size is backward-compatible with
		//        .    previous LAPACK versions. ====
		work.Set(0, math.Max(float64(max(1, n)), work.Get(0)))
	}

	return
}
