;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Percolation Model of Innovation in Complex Technology Spaces
; After the papers by Silverberg & Verspagen (2005; 2007)
; This version (C) Christopher J Watts, 2011
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

extensions [array]

globals [
  num-columns
  bpf
  baseline
  
  patchqueue ; Array used for calculating paths. (Lists are inefficient as queues.)
  max-q-size ; Trouble with arrays is: we have to pre-state their max length.
  
  innov-sizes
  freq-distrib
  
  state-freq
  mean-height
  mean-pathlength
  
  num-reachable
  num-possible
  perc-reachable
  
  num-recent-changes
  num-changes
  deadlocked
  
  state-colors
]

patches-own [
  state
  bp-frontier
  pathlength
  tempdist
  diamond-neighbors
  visited
  reachable
]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
to setup
  clear-all
  
  set state-colors array:from-list (list black white yellow pink green grey)
  
  set num-columns 1 + max-pxcor - min-pxcor
  set max-q-size 1 + ((2 * search-radius-m) * (search-radius-m + 1))
  if (2 * num-columns) > max-q-size [set max-q-size (2 * num-columns)]
  set patchqueue array:from-list n-values max-q-size [nobody]
  set state-freq array:from-list n-values 4 [0]
  set bpf array:from-list n-values num-columns [nobody]
  set innov-sizes []
    
  ask patches [
    set tempdist -1
    set diamond-neighbors []
    ifelse chance-possible-q > random 100 [
      set state 1
      set pcolor array:item state-colors 1
    ]
    [
      set state 0
      set pcolor array:item state-colors 0
    ]
  ]
  
  set baseline patches with [pycor = 0]
  ask baseline [
    set state 3
    set pcolor array:item state-colors 3
  ]
  set num-possible count patches with [(state > 0) and (pycor > 0)]
  ask patches with [pycor > 0] [array:set state-freq state (1 + array:item state-freq state)]
  
  calc-pathlength-via-possibles
  calc-pathlength
  ask patches with [state = 3] [update-bpf]
  
  setup-plots
  update-plots
  
  set deadlocked false
  set num-changes 0
end

to calc-one-neighborhood
  ; Calculates a diamond shape of patches, m in radius, excluding the centre patch (caller).
  let diamond []
  let cur-site nobody
  let q-start 0
  let q-size 0
  
  ; Add to queue
  array:set patchqueue (q-start + q-size) self
  set q-size (q-size + 1)
  set tempdist 0
  while [q-size > 0] [
    ; Take from start of queue.
    set q-size q-size - 1
    set cur-site array:item patchqueue q-start
    set q-start (q-start + 1)
    
    ask cur-site [
      if tempdist < search-radius-m [
        ask neighbors4 with [tempdist = -1] [
          set tempdist (1 + [tempdist] of myself)
          set diamond fput self diamond
          ; Add to end of queue
          array:set patchqueue (q-start + q-size) self
          set q-size (q-size + 1)
        ]
      ]
    ]
  ]
  foreach diamond [ ask ? [ set tempdist -1 ] ] ; Clean up
  set tempdist -1
  set diamond-neighbors diamond
end

to calc-pathlength-via-possibles
  ; Calculates minimum degree of separation from baseline
  ; via valid path, where valid paths consist only of neighbors4 in state >= 1.
  let cur-pathlength 0
  let q-start 0
  let q-size 0
  let cur-site nobody
  ask patches [set pathlength -1]
  ask baseline [
    array:set patchqueue ((q-start + q-size) mod max-q-size) self
    set q-size ((q-size + 1) mod max-q-size) ; Will do bad things, if q-size > max-q-size - i.e. we loop round on ourselves!
    set pathlength 0
  ]
  while [q-size > 0] [
    ; Take from start of queue.
    if q-size >= max-q-size [
      user-message (word "Queue is too big!")
      stop
    ]
    set q-size q-size - 1
    set cur-site array:item patchqueue q-start
    set q-start (q-start + 1) mod max-q-size
    
    ask cur-site [
      ;print (word self ": " pathlength) ;; For debugging
      set cur-pathlength pathlength
      ask neighbors4 with [state >= 1] [
        if pathlength = -1 [ ; Implies we haven't been here before.
          set pathlength cur-pathlength + 1
          ; Add to end of queue
          array:set patchqueue ((q-start + q-size) mod max-q-size) self
          set q-size ((q-size + 1) mod max-q-size)
        ]
      ]
    ]
  ]
  ask patches [set reachable ((pycor > 0) and (state > 0) and (pathlength != -1))]
  set num-reachable count patches with [reachable]
  set perc-reachable ifelse-value (num-possible = 0) [0] [num-reachable / num-possible]
  highlight-unreachables
end

to highlight-unreachables
  ask patches with [state > 0 and not reachable] [set pcolor array:item state-colors 5]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
to go
  if ticks >= max-ticks [stop]
  if deadlocked [stop]
  
  search-from-bpf
  
  calc-pathlength
  tick
  update-plots
  
  
end

to search-from-bpf
  let cur-diamond []
  let test-chance 0
  
  foreach (array:to-list bpf) [
    ask ? [
      ;print diamond-neighbors ;; For debugging
      if 0 = (length diamond-neighbors) [calc-one-neighborhood]
      set test-chance search-effort-e / (length diamond-neighbors)
      ;print test-chance
      foreach diamond-neighbors [
        ask ? [
          ;  if state = 1 [ ; Possible. We don't bother counting the search for the impossible.
          ;    if reachable [ ; Only bother with reachable technologies.
          if (state = 1) [if reachable [search-site test-chance]]
        ]
      ]
    ]
  ]
end

to search-site [test-chance]
  if test-chance > random-float 1 [ ; Lucky?
    ; Discovered!
    set state 2
    set pcolor array:item state-colors 2
    array:set state-freq 2 (1 + array:item state-freq 2)
    array:set state-freq 1 (-1 + array:item state-freq 1)
    set num-changes num-changes + 1
    stop
  ]
end

to calc-pathlength
  ; Calculates minimum degree of separation from baseline
  ; via valid path, where valid paths consist only of neighbors4 in state 3.
  let cur-pathlength 0
  let q-start 0
  let q-size 0
  let cur-site nobody
  ask patches [set pathlength -1]
  ask baseline [
    array:set patchqueue ((q-start + q-size) mod max-q-size) self
    set q-size ((q-size + 1) mod max-q-size) ; NB. Will do bad things, if q-size > max-q-size - i.e. we'll loop round on ourselves!
    set pathlength 0
  ]
  while [q-size > 0] [
    ; Take from start of queue.
    if q-size >= max-q-size [ ; Error check
      user-message (word "Queue is too big!")
      stop
    ]
    set q-size q-size - 1
    set cur-site array:item patchqueue q-start
    set q-start (q-start + 1) mod max-q-size
    
    ask cur-site [
      ;print (word self ": " pathlength) ;; For debugging
      set cur-pathlength pathlength
      ask neighbors4 with [pathlength = -1] [ ; Implies we haven't been here before.
        if state >= 2 [ 
;      ask neighbors4 with [state >= 2] [
;        if pathlength = -1 [ ; Implies we haven't been here before.
          set pathlength cur-pathlength + 1
          if state = 2 [
            set state 3
            set pcolor array:item state-colors 3
            array:set state-freq 3 (1 + array:item state-freq 3)
            array:set state-freq 2 (-1 + array:item state-freq 2)
            set num-changes num-changes + 1
            update-bpf
          ]
          ; Add to end of queue
          array:set patchqueue ((q-start + q-size) mod max-q-size) self
          set q-size ((q-size + 1) mod max-q-size)
        ]
      ]
    ]
  ]
end

to update-bpf
  ; Recalculates the "best-practice frontier" in current patch's column
  ; BPF is in each column the highest point in state 3
  
  ifelse (array:item bpf pxcor) = nobody [
    array:set bpf pxcor self
    set pcolor array:item state-colors 4
  ]
  [
    if pycor > [pycor] of (array:item bpf pxcor) [
      ask (array:item bpf pxcor) [set pcolor array:item state-colors 3]
      set innov-sizes fput (pycor - [pycor] of (array:item bpf pxcor)) innov-sizes
      array:set bpf pxcor self
      set pcolor array:item state-colors 4
    ]
  ]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
to setup-plots
  set-current-plot "Progress"
  set-current-plot-pen "BPF Height"
  set-plot-pen-interval output-every
  set-current-plot-pen "Path Length"
  set-plot-pen-interval output-every
end

to update-plots
  if 0 = ticks mod output-every [
    set mean-height mean map [[pycor] of ?] array:to-list bpf
    set mean-pathlength mean [pathlength] of patches with [state = 3]
    
    set-current-plot "States"
    clear-plot
    let cur-state 0
    repeat 4 [
      plot array:item state-freq cur-state
      set cur-state cur-state + 1
    ]
    
    set-current-plot "Progress"
    set-current-plot-pen "BPF Height"
    plot mean-height
    set-current-plot-pen "Path Length"
    plot mean-pathlength
    
    if 0 < length innov-sizes [
      set freq-distrib []
      set-current-plot "Log-Log"
      clear-plot
      set innov-sizes sort innov-sizes
      let cur-val first innov-sizes
      let cur-freq 1
      foreach but-first innov-sizes [
        ifelse ? = cur-val [
          set cur-freq cur-freq + 1
        ]
        [
          set freq-distrib fput (list cur-val cur-freq) freq-distrib
          plotxy (log cur-val 10) (log cur-freq 10)
          set cur-val ?
          set cur-freq 1
        ]
      ]
      plotxy (log cur-val 10) (log cur-freq 10)
      set freq-distrib fput (list cur-val cur-freq) freq-distrib

      set-current-plot "Innovation Sizes"
      set-plot-x-range 0 (1 + max innov-sizes)
      histogram innov-sizes
    ]
    set num-recent-changes num-changes
    set deadlocked (num-changes = 0)
    set num-changes 0
  ]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
to test-time
  setup
  let start-time timer
  repeat 1000 [go]
  print (word "Time taken: " (timer - start-time))
end