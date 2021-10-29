package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Ztprfb applies a complex "triangular-pentagonal" block reflector H or its
// conjugate transpose H**H to a complex matrix C, which is composed of two
// blocks A and B, either from the left or right.
func Ztprfb(side mat.MatSide, trans mat.MatTrans, direct, storev byte, m, n, k, l int, v, t, a, b, work *mat.CMatrix) (err error) {
	var backward, column, forward, left, right, row bool
	var one, zero complex128
	var i, j, kp, mp, np int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Quick return if possible
	if m <= 0 || n <= 0 || k <= 0 || l < 0 {
		return
	}

	if storev == 'C' {
		column = true
		row = false
	} else if storev == 'R' {
		column = false
		row = true
	} else {
		column = false
		row = false
	}

	if side == Left {
		left = true
		right = false
	} else if side == Right {
		left = false
		right = true
	} else {
		left = false
		right = false
	}

	if direct == 'F' {
		forward = true
		backward = false
	} else if direct == 'B' {
		forward = false
		backward = true
	} else {
		forward = false
		backward = false
	}

	// ---------------------------------------------------------------------------
	if column && forward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I ]    (K-by-K)
		//                  [ V ]    (M-by-K)
		//
		//        Form  H C  or  H**H C  where  C = [ A ]  (K-by-N)
		//                                          [ B ]  (M-by-N)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A -   T (A + V**H B)  or  A = A -   T**H (A + V**H B)
		//        B = B - V T (A + V**H B)  or  B = B - V T**H (A + V**H B)
		//
		// ---------------------------------------------------------------------------
		mp = min(m-l+1, m)
		kp = min(l+1, k)

		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				work.Set(i-1, j-1, b.Get(m-l+i-1, j-1))
			}
		}
		if err = goblas.Ztrmm(Left, Upper, ConjTrans, NonUnit, l, n, one, v.Off(mp-1, 0), work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(ConjTrans, NoTrans, l, n, m-l, one, v, b, one, work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(ConjTrans, NoTrans, k-l, n, m, one, v.Off(0, kp-1), b, zero, work.Off(kp-1, 0)); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Left, Upper, trans, NonUnit, k, n, one, t, work); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(NoTrans, NoTrans, m-l, n, k, -one, v, work, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, l, n, k-l, -one, v.Off(mp-1, kp-1), work.Off(kp-1, 0), one, b.Off(mp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Left, Upper, NoTrans, NonUnit, l, n, one, v.Off(mp-1, 0), work); err != nil {
			panic(err)
		}
		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				b.Set(m-l+i-1, j-1, b.Get(m-l+i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if column && forward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I ]    (K-by-K)
		//                  [ V ]    (N-by-K)
		//
		//        Form  C H or  C H**H  where  C = [ A B ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A - (A + B V) T      or  A = A - (A + B V) T**H
		//        B = B - (A + B V) T V**H  or  B = B - (A + B V) T**H V**H
		//
		// ---------------------------------------------------------------------------
		np = min(n-l+1, n)
		kp = min(l+1, k)

		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, j-1, b.Get(i-1, n-l+j-1))
			}
		}
		if err = goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, m, l, one, v.Off(np-1, 0), work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, m, l, n-l, one, b, v, one, work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, m, k-l, n, one, b, v.Off(0, kp-1), zero, work.Off(0, kp-1)); err != nil {
			panic(err)
		}

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		err = goblas.Ztrmm(Right, Upper, trans, NonUnit, m, k, one, t, work)

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(NoTrans, ConjTrans, m, n-l, k, -one, work, v, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, l, k-l, -one, work.Off(0, kp-1), v.Off(np-1, kp-1), one, b.Off(0, np-1)); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Right, Upper, ConjTrans, NonUnit, m, l, one, v.Off(np-1, 0), work); err != nil {
			panic(err)
		}
		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				b.Set(i-1, n-l+j-1, b.Get(i-1, n-l+j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if column && backward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V ]    (M-by-K)
		//                  [ I ]    (K-by-K)
		//
		//        Form  H C  or  H**H C  where  C = [ B ]  (M-by-N)
		//                                          [ A ]  (K-by-N)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A -   T (A + V**H B)  or  A = A -   T**H (A + V**H B)
		//        B = B - V T (A + V**H B)  or  B = B - V T**H (A + V**H B)
		//
		// ---------------------------------------------------------------------------
		mp = min(l+1, m)
		kp = min(k-l+1, k)

		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				work.Set(k-l+i-1, j-1, b.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Left, Lower, ConjTrans, NonUnit, l, n, one, v.Off(0, kp-1), work.Off(kp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(ConjTrans, NoTrans, l, n, m-l, one, v.Off(mp-1, kp-1), b.Off(mp-1, 0), one, work.Off(kp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(ConjTrans, NoTrans, k-l, n, m, one, v, b, zero, work); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Left, Lower, trans, NonUnit, k, n, one, t, work); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(NoTrans, NoTrans, m-l, n, k, -one, v.Off(mp-1, 0), work, one, b.Off(mp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, l, n, k-l, -one, v, work, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Left, Lower, NoTrans, NonUnit, l, n, one, v.Off(0, kp-1), work.Off(kp-1, 0)); err != nil {
			panic(err)
		}
		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(k-l+i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if column && backward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V ]    (N-by-K)
		//                  [ I ]    (K-by-K)
		//
		//        Form  C H  or  C H**H  where  C = [ B A ] (B is M-by-N, A is M-by-K)
		//
		//        H = I - W T W**H          or  H**H = I - W T**H W**H
		//
		//        A = A - (A + B V) T      or  A = A - (A + B V) T**H
		//        B = B - (A + B V) T V**H  or  B = B - (A + B V) T**H V**H
		//
		// ---------------------------------------------------------------------------
		np = min(l+1, n)
		kp = min(k-l+1, k)

		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, k-l+j-1, b.Get(i-1, j-1))
			}
		}
		if err = goblas.Ztrmm(Right, Lower, NoTrans, NonUnit, m, l, one, v.Off(0, kp-1), work.Off(0, kp-1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, m, l, n-l, one, b.Off(0, np-1), v.Off(np-1, kp-1), one, work.Off(0, kp-1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, m, k-l, n, one, b, v, zero, work); err != nil {
			panic(err)
		}

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Right, Lower, trans, NonUnit, m, k, one, t, work); err != nil {
			panic(err)
		}

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(NoTrans, ConjTrans, m, n-l, k, -one, work, v.Off(np-1, 0), one, b.Off(0, np-1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, l, k-l, -one, work, v, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Right, Lower, ConjTrans, NonUnit, m, l, one, v.Off(0, kp-1), work.Off(0, kp-1)); err != nil {
			panic(err)
		}
		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(i-1, k-l+j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && forward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I V ] ( I is K-by-K, V is K-by-M )
		//
		//        Form  H C  or  H**H C  where  C = [ A ]  (K-by-N)
		//                                          [ B ]  (M-by-N)
		//
		//        H = I - W**H T W          or  H**H = I - W**H T**H W
		//
		//        A = A -     T (A + V B)  or  A = A -     T**H (A + V B)
		//        B = B - V**H T (A + V B)  or  B = B - V**H T**H (A + V B)
		//
		// ---------------------------------------------------------------------------
		mp = min(m-l+1, m)
		kp = min(l+1, k)

		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				work.Set(i-1, j-1, b.Get(m-l+i-1, j-1))
			}
		}
		if err = goblas.Ztrmm(Left, Lower, NoTrans, NonUnit, l, n, one, v.Off(0, mp-1), work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, l, n, m-l, one, v, b, one, work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, k-l, n, m, one, v.Off(kp-1, 0), b, zero, work.Off(kp-1, 0)); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Left, Upper, trans, NonUnit, k, n, one, t, work); err != nil {
			panic(err)
		}
		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(ConjTrans, NoTrans, m-l, n, k, -one, v, work, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(ConjTrans, NoTrans, l, n, k-l, -one, v.Off(kp-1, mp-1), work.Off(kp-1, 0), one, b.Off(mp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Left, Lower, ConjTrans, NonUnit, l, n, one, v.Off(0, mp-1), work); err != nil {
			panic(err)
		}
		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				b.Set(m-l+i-1, j-1, b.Get(m-l+i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && forward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ I V ] ( I is K-by-K, V is K-by-N )
		//
		//        Form  C H  or  C H**H  where  C = [ A B ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W**H T W            or  H**H = I - W**H T**H W
		//
		//        A = A - (A + B V**H) T      or  A = A - (A + B V**H) T**H
		//        B = B - (A + B V**H) T V    or  B = B - (A + B V**H) T**H V
		//
		// ---------------------------------------------------------------------------
		np = min(n-l+1, n)
		kp = min(l+1, k)

		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, j-1, b.Get(i-1, n-l+j-1))
			}
		}
		if err = goblas.Ztrmm(Right, Lower, ConjTrans, NonUnit, m, l, one, v.Off(0, np-1), work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, l, n-l, one, b, v, one, work); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, k-l, n, one, b, v.Off(kp-1, 0), zero, work.Off(0, kp-1)); err != nil {
			panic(err)
		}

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Right, Upper, trans, NonUnit, m, k, one, t, work); err != nil {
			panic(err)
		}

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(NoTrans, NoTrans, m, n-l, k, -one, work, v, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, m, l, k-l, -one, work.Off(0, kp-1), v.Off(kp-1, np-1), one, b.Off(0, np-1)); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Right, Lower, NoTrans, NonUnit, m, l, one, v.Off(0, np-1), work); err != nil {
			panic(err)
		}
		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				b.Set(i-1, n-l+j-1, b.Get(i-1, n-l+j-1)-work.Get(i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && backward && left {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V I ] ( I is K-by-K, V is K-by-M )
		//
		//        Form  H C  or  H**H C  where  C = [ B ]  (M-by-N)
		//                                          [ A ]  (K-by-N)
		//
		//        H = I - W**H T W          or  H**H = I - W**H T**H W
		//
		//        A = A -     T (A + V B)  or  A = A -     T**H (A + V B)
		//        B = B - V**H T (A + V B)  or  B = B - V**H T**H (A + V B)
		//
		// ---------------------------------------------------------------------------
		mp = min(l+1, m)
		kp = min(k-l+1, k)

		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				work.Set(k-l+i-1, j-1, b.Get(i-1, j-1))
			}
		}
		if err = goblas.Ztrmm(Left, Upper, NoTrans, NonUnit, l, n, one, v.Off(kp-1, 0), work.Off(kp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, l, n, m-l, one, v.Off(kp-1, mp-1), b.Off(mp-1, 0), one, work.Off(kp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, k-l, n, m, one, v, b, zero, work); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Left, Lower, trans, NonUnit, k, n, one, t, work); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			for i = 1; i <= k; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(ConjTrans, NoTrans, m-l, n, k, -one, v.Off(0, mp-1), work, one, b.Off(mp-1, 0)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(ConjTrans, NoTrans, l, n, k-l, -one, v, work, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Left, Upper, ConjTrans, NonUnit, l, n, one, v.Off(kp-1, 0), work.Off(kp-1, 0)); err != nil {
			panic(err)
		}
		for j = 1; j <= n; j++ {
			for i = 1; i <= l; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(k-l+i-1, j-1))
			}
		}

		// ---------------------------------------------------------------------------
	} else if row && backward && right {
		// ---------------------------------------------------------------------------
		//
		//        Let  W =  [ V I ] ( I is K-by-K, V is K-by-N )
		//
		//        Form  C H  or  C H**H  where  C = [ B A ] (A is M-by-K, B is M-by-N)
		//
		//        H = I - W**H T W            or  H**H = I - W**H T**H W
		//
		//        A = A - (A + B V**H) T      or  A = A - (A + B V**H) T**H
		//        B = B - (A + B V**H) T V    or  B = B - (A + B V**H) T**H V
		//
		// ---------------------------------------------------------------------------
		np = min(l+1, n)
		kp = min(k-l+1, k)

		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, k-l+j-1, b.Get(i-1, j-1))
			}
		}
		if err = goblas.Ztrmm(Right, Upper, ConjTrans, NonUnit, m, l, one, v.Off(kp-1, 0), work.Off(0, kp-1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, l, n-l, one, b.Off(0, np-1), v.Off(kp-1, np-1), one, work.Off(0, kp-1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, ConjTrans, m, k-l, n, one, b, v, zero, work); err != nil {
			panic(err)
		}

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				work.Set(i-1, j-1, work.Get(i-1, j-1)+a.Get(i-1, j-1))
			}
		}

		if err = goblas.Ztrmm(Right, Lower, trans, NonUnit, m, k, one, t, work); err != nil {
			panic(err)
		}

		for j = 1; j <= k; j++ {
			for i = 1; i <= m; i++ {
				a.Set(i-1, j-1, a.Get(i-1, j-1)-work.Get(i-1, j-1))
			}
		}

		if err = goblas.Zgemm(NoTrans, NoTrans, m, n-l, k, -one, work, v.Off(0, np-1), one, b.Off(0, np-1)); err != nil {
			panic(err)
		}
		if err = goblas.Zgemm(NoTrans, NoTrans, m, l, k-l, -one, work, v, one, b); err != nil {
			panic(err)
		}
		if err = goblas.Ztrmm(Right, Upper, NoTrans, NonUnit, m, l, one, v.Off(kp-1, 0), work.Off(0, kp-1)); err != nil {
			panic(err)
		}
		for j = 1; j <= l; j++ {
			for i = 1; i <= m; i++ {
				b.Set(i-1, j-1, b.Get(i-1, j-1)-work.Get(i-1, k-l+j-1))
			}
		}

	}

	return
}
